import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.signal
import wfdb

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watch_lsm.models.backbone import WatchLSMModel
from watch_lsm.models.heads import WatchLSMDownstreamModel
from watch_lsm.data.dataset import SmartwatchDataset
from watch_lsm.paths import data_path, model_path, project_path, machine_config
from watch_lsm.runtime import select_device, seed_everything

def load_downstream_data():
    records = [
        ("walk", "datasets/wrist/s1_walk", 1),
        ("walk", "datasets/wrist/s2_walk", 1),
        ("run", "datasets/wrist/s3_run", 2),
        ("run", "datasets/wrist/s5_run", 2),
        ("bike", "datasets/wrist/s1_low_resistance_bike", 3)
    ]
    
    ppg_list = []
    acc_list = []
    hr_list = []
    cad_list = []
    act_list = []
    
    for act_name, r, act_id in records:
        r = str(data_path("wrist", Path(r).name))
        if os.path.exists(r + ".hea") and os.path.exists(r + ".dat"):
            rec = wfdb.rdrecord(r)
            fs = rec.fs
            df = pd.DataFrame(rec.p_signal).ffill().bfill().values
            df = df[int(2.0 * fs):] # Skip warmup
            
            ecg = df[:, 0]
            ppg = df[:, 1]
            acc = df[:, 5:8]
            
            # Compute reference HR from ECG (1 sample per second)
            b, a = scipy.signal.butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype='bandpass')
            ecg_filt = scipy.signal.filtfilt(b, a, ecg)
            peaks, _ = scipy.signal.find_peaks(ecg_filt**2, distance=int(0.3 * fs), height=np.mean(ecg_filt**2))
            
            duration_s = int(len(ecg) / fs)
            hr_series = []
            cad_series = []
            
            acc_mag = np.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
            
            for s in range(duration_s):
                t_start = s * fs
                t_end = (s + 8) * fs # 8-second window
                win_peaks = peaks[(peaks >= t_start) & (peaks < t_end)]
                if len(win_peaks) >= 2:
                    hr = 60.0 / np.median(np.diff(win_peaks) / fs)
                else:
                    hr = hr_series[-1] if len(hr_series) > 0 else 75.0
                hr_series.append(hr)
                
                # Cadence from ACC FFT
                win_acc = acc_mag[t_start:t_end]
                fft_vals = np.abs(np.fft.rfft(win_acc - np.mean(win_acc)))
                freqs = np.fft.rfftfreq(len(win_acc), d=1/fs)
                m = (freqs >= 0.8) & (freqs <= 3.5)
                cad = freqs[m][np.argmax(fft_vals[m])] * 60.0 if np.any(m) else 0.0
                cad_series.append(cad)
                
            ppg_list.append(ppg)
            acc_list.append(acc)
            hr_list.append(hr_series)
            cad_list.append(cad_series)
            act_list.append(act_id)
            
    return ppg_list, acc_list, hr_list, cad_list, act_list

def main():
    parser = argparse.ArgumentParser(description="Watch-LSM Downstream Fine-Tuning & Evaluation")
    parser.add_argument("--task", type=str, default="heart_rate", choices=["heart_rate", "cadence", "activity"])
    parser.add_argument("--freeze_backbone", action="store_true", help="Linear probing mode (freeze backbone)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrained_ckpt", type=str)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from_scratch", action="store_true", help="Explicitly allow training without a checkpoint")
    args = parser.parse_args()

    device = select_device(args.device)
    runtime = machine_config().get("runtime", {})
    args.batch_size = args.batch_size if args.batch_size is not None else runtime.get("batch_size", 8 if device.type == "cpu" else 32)
    args.num_workers = args.num_workers if args.num_workers is not None else runtime.get("num_workers", 0)
    args.pretrained_ckpt = project_path(args.pretrained_ckpt) if args.pretrained_ckpt else model_path("checkpoints", "pretrain", "best_foundation_model.pt")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0:
        parser.error("epochs/batch_size/lr must be positive and num_workers nonnegative")
    seed_everything(args.seed)
    print(f"Task: {args.task.upper()} | Linear Probe (Freeze): {args.freeze_backbone} | Device: {device}")

    # Load data
    ppg_list, acc_list, hr_list, cad_list, act_list = load_downstream_data()
    if len(ppg_list) != 5:
        raise RuntimeError("Expected all 5 wrist recordings. Check configured datasets directory.")
    train_ppg, train_acc = ppg_list[:3], acc_list[:3]
    val_ppg, val_acc = ppg_list[3:], acc_list[3:]

    train_hr, val_hr = (hr_list[:3], hr_list[3:]) if args.task == "heart_rate" else (None, None)
    train_cad, val_cad = (cad_list[:3], cad_list[3:]) if args.task == "cadence" else (None, None)
    train_act, val_act = (act_list[:3], act_list[3:]) if args.task == "activity" else (None, None)

    train_ds = SmartwatchDataset(train_ppg, train_acc, hr_list=train_hr, cadence_list=train_cad, activity_list=train_act, is_training=True, augment=True)
    val_ds = SmartwatchDataset(val_ppg, val_acc, hr_list=val_hr, cadence_list=val_cad, activity_list=val_act, is_training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Base Foundation Model
    foundation = WatchLSMModel(
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        ppg_patch_size=16,
        acc_patch_size=8,
        num_patches=32
    )

    if os.path.exists(args.pretrained_ckpt) and not args.from_scratch:
        print(f"Loading pretrained backbone weights from {args.pretrained_ckpt}...")
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=True)
        if "model_config" in ckpt:
            foundation = WatchLSMModel(**ckpt["model_config"])
        foundation.load_state_dict(ckpt["model_state_dict"])
    elif args.from_scratch:
        print("Explicit --from_scratch: using random initialization.")
    else:
        raise FileNotFoundError(f"Checkpoint missing: {args.pretrained_ckpt}. Use --from_scratch explicitly to train without it.")

    # Downstream Model
    model = WatchLSMDownstreamModel(
        foundation_model=foundation,
        task=args.task,
        num_classes=4,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = torch.nn.L1Loss() if args.task in ["heart_rate", "cadence"] else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)

    print(f"\n--- Training Downstream Head for {args.task.upper()} ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            x_p = batch["ppg"].to(device)
            x_a = batch["acc"].to(device)
            
            if args.task == "heart_rate":
                target = batch["hr"].to(device)
            elif args.task == "cadence":
                target = batch["cadence"].to(device)
            else:
                target = batch["activity"].to(device)
                
            optimizer.zero_grad()
            pred = model(x_p, x_a)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        total_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_errors = []
        with torch.no_grad():
            for batch in val_loader:
                x_p = batch["ppg"].to(device)
                x_a = batch["acc"].to(device)
                if args.task == "heart_rate":
                    target = batch["hr"].to(device)
                    pred = model(x_p, x_a)
                    val_errors.extend(torch.abs(pred - target).cpu().numpy())
                elif args.task == "cadence":
                    target = batch["cadence"].to(device)
                    pred = model(x_p, x_a)
                    val_errors.extend(torch.abs(pred - target).cpu().numpy())
                else:
                    target = batch["activity"].to(device)
                    pred = model(x_p, x_a).argmax(dim=-1)
                    val_errors.extend((pred == target).cpu().numpy())
                    
        val_metric = np.mean(val_errors)
        metric_name = "MAE (BPM)" if args.task == "heart_rate" else ("MAE (SPM)" if args.task == "cadence" else "Accuracy")
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {total_loss:.4f} | Val {metric_name}: {val_metric:.2f}")

    print(f"\nDownstream evaluation for {args.task} finished! Final Val {metric_name}: {val_metric:.2f}")

if __name__ == "__main__":
    main()
