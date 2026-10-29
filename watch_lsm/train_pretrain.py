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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watch_lsm.models.backbone import WatchLSMModel
from watch_lsm.models.losses import CompositePretrainLoss
from watch_lsm.data.dataset import SmartwatchDataset
from watch_lsm.paths import data_path, model_path, project_path, load_yaml, machine_config
from watch_lsm.runtime import select_device, seed_everything

def load_data_from_wrist():
    records = [
        str(data_path("wrist", "s1_walk")),
        str(data_path("wrist", "s2_walk")),
        str(data_path("wrist", "s3_run")),
        str(data_path("wrist", "s5_run")),
        str(data_path("wrist", "s1_low_resistance_bike"))
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    training = config["training"]
    runtime = machine_config().get("runtime", {})
    device = select_device(args.device)
    args.epochs = args.epochs if args.epochs is not None else training["max_epochs"]
    args.batch_size = args.batch_size if args.batch_size is not None else runtime.get("batch_size", 8 if device.type == "cpu" else training["batch_size"])
    args.lr = args.lr if args.lr is not None else training["learning_rate"]
    args.num_workers = args.num_workers if args.num_workers is not None else runtime.get("num_workers", 0)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("epochs/batch_size/lr must be positive and num_workers nonnegative")
    configured_save = Path(training.get("save_dir", "models/checkpoints/pretrain"))
    if args.save_dir:
        args.save_dir = project_path(args.save_dir)
    elif not configured_save.is_absolute() and configured_save.parts[0] == "models":
        args.save_dir = model_path(*configured_save.parts[1:])
    else:
        args.save_dir = project_path(configured_save)
    seed_everything(config.get("seed", 42))
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Load data
    print("Loading smartwatch pretraining recordings...")
    ppg_list, acc_list = load_data_from_wrist()
    if len(ppg_list) != 5:
        raise RuntimeError("Expected all 5 wrist recordings. Check the configured datasets directory.")
    print(f"Loaded {len(ppg_list)} long recording sessions.")

    # Split train / val
    train_ppg = ppg_list[:3]
    train_acc = acc_list[:3]
    val_ppg = ppg_list[3:]
    val_acc = acc_list[3:]

    signals = config["signals"]
    dataset_args = dict(window_sec=signals["window_duration_sec"], target_fs_ppg=signals["ppg"]["sampling_rate"], target_fs_acc=signals["acc"]["sampling_rate"], step_sec=2.0)
    train_dataset = SmartwatchDataset(train_ppg, train_acc, **dataset_args, augment=True)
    val_dataset = SmartwatchDataset(val_ppg, val_acc, **dataset_args, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model initialization
    patch_sec = signals["ppg"]["patch_len_sec"]
    if patch_sec != signals["acc"]["patch_len_sec"]:
        raise ValueError("PPG and ACC patch durations must match")
    num_patches = int(signals["window_duration_sec"] / patch_sec)
    model_config = {key: config["model"][key] for key in ["hidden_dim", "n_layers", "n_heads", "dropout"]}
    model_config.update(ppg_patch_size=int(patch_sec * signals["ppg"]["sampling_rate"]), acc_patch_size=int(patch_sec * signals["acc"]["sampling_rate"]), num_patches=num_patches)
    model = WatchLSMModel(**model_config).to(device)

    weights = config["ssl"]["loss_weights"]
    criterion = CompositePretrainLoss(w_recon=weights["mae_recon"], w_cross=weights["cross_modal_infonce"], w_spectral=weights["spectral_fft"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=training["weight_decay"])
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
            N = num_patches
            mask_p = (torch.rand(B, N, device=device) < config["ssl"]["mask_ratio_ppg"]).float()
            mask_a = (torch.rand(B, N, device=device) < config["ssl"]["mask_ratio_acc"]).float()
            
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
                mask_p = (torch.rand(B, N, device=device) < config["ssl"]["mask_ratio_ppg"]).float()
                mask_a = (torch.rand(B, N, device=device) < config["ssl"]["mask_ratio_acc"]).float()
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
                "val_loss": val_loss,
                "model_config": model_config,
                "config": config,
                "device": str(device)
            }, ckpt_path)
            print(f"  -> Saved best foundation model checkpoint to {ckpt_path}")

    print("\nPretraining completed successfully!")

if __name__ == "__main__":
    main()
