import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wfdb

# Add current dir to path
sys.path.append(os.path.abspath("."))
from watch_lsm.models.backbone import WatchLSMModel
from watch_lsm.models.losses import CompositePretrainLoss
from watch_lsm.data.dataset import SmartwatchDataset

def load_data_from_wrist():
    records = [
        "datasets/wrist/s1_walk",
        "datasets/wrist/s2_walk",
        "datasets/wrist/s3_run",
        "datasets/wrist/s5_run",
        "datasets/wrist/s1_low_resistance_bike"
    ]
    ppg_list = []
    acc_list = []
    
    for r in records:
        if os.path.exists(r + ".hea") and os.path.exists(r + ".dat"):
            rec = wfdb.rdrecord(r)
            fs = rec.fs
            df = pd.DataFrame(rec.p_signal).ffill().bfill().values
            # Warm up
            df = df[int(2.0 * fs):]
            ppg = df[:, 1]
            acc = df[:, 5:8]
            ppg_list.append(ppg)
            acc_list.append(acc)
            
    return ppg_list, acc_list

def main():
    parser = argparse.ArgumentParser(description="Watch-LSM Self-Supervised Pretraining")
    parser.add_argument("--config", type=str, default="watch_lsm/configs/pretrain_config.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--save_dir", type=str, default="models/checkpoints/pretrain")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading smartwatch pretraining recordings...")
    ppg_list, acc_list = load_data_from_wrist()
    print(f"Loaded {len(ppg_list)} long recording sessions.")

    # Split train / val
    train_ppg = ppg_list[:3]
    train_acc = acc_list[:3]
    val_ppg = ppg_list[3:]
    val_acc = acc_list[3:]

    train_dataset = SmartwatchDataset(train_ppg, train_acc, window_sec=8.0, step_sec=2.0, augment=True)
    val_dataset = SmartwatchDataset(val_ppg, val_acc, window_sec=8.0, step_sec=2.0, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model initialization
    model = WatchLSMModel(
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        ppg_patch_size=16, # 0.25s at 64Hz
        acc_patch_size=8,   # 0.25s at 32Hz
        num_patches=32
    ).to(device)

    criterion = CompositePretrainLoss(w_recon=1.0, w_cross=0.2, w_spectral=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    print("\n--- Starting Watch-LSM Foundation Model Pretraining ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_cross = 0.0
        
        for batch in train_loader:
            x_p = batch["ppg"].to(device) # (B, 1, 512)
            x_a = batch["acc"].to(device) # (B, 3, 256)
            
            # Create random patch masks
            B = x_p.size(0)
            N = 32
            mask_p = (torch.rand(B, N, device=device) < 0.4).float()
            mask_a = (torch.rand(B, N, device=device) < 0.3).float()
            
            optimizer.zero_grad()
            outputs = model(x_p, x_a, mask_ppg=mask_p, mask_acc=mask_a)
            loss_dict = criterion(outputs, mask_ppg=mask_p, mask_acc=mask_a)
            
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += loss_dict["recon_loss"].item()
            train_cross += loss_dict["cross_modal_loss"].item()
            
        scheduler.step()
        n_batches = len(train_loader)
        train_loss /= n_batches
        train_recon /= n_batches
        train_cross /= n_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_p = batch["ppg"].to(device)
                x_a = batch["acc"].to(device)
                B = x_p.size(0)
                mask_p = (torch.rand(B, N, device=device) < 0.4).float()
                mask_a = (torch.rand(B, N, device=device) < 0.3).float()
                outputs = model(x_p, x_a, mask_ppg=mask_p, mask_acc=mask_a)
                val_losses = criterion(outputs, mask_ppg=mask_p, mask_acc=mask_a)
                val_loss += val_losses["total_loss"].item()
        val_loss /= max(len(val_loader), 1)
        
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, Cross: {train_cross:.4f}) | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, "best_foundation_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, ckpt_path)
            print(f"  -> Saved best foundation model checkpoint to {ckpt_path}")

    print("\nPretraining completed successfully!")

if __name__ == "__main__":
    main()
