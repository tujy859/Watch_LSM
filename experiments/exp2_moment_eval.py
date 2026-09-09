import torch
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import json
import os
from momentfm import MOMENTPipeline

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

print("Initializing MOMENT-1-small for reconstruction task...")
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small",
    model_kwargs={"task_name": "reconstruction"}
)
model.init()
model.eval()

# Load smartwatch signals
rec_walk = wfdb.rdrecord("datasets/wrist/s1_walk")
rec_run = wfdb.rdrecord("datasets/wrist/s3_run")

fs = rec_walk.fs # 256 Hz
# Resample or downsample to 512 sequence length for MOMENT
# 512 samples at 64 Hz = 8 seconds
target_seq_len = 512

def prepare_segments(rec, start_sec=30):
    fs = rec.fs
    # Take 8 seconds
    s_start = int(start_sec * fs)
    s_end = s_start + int(8 * fs)
    
    ppg = rec.p_signal[s_start:s_end, 1]
    acc = rec.p_signal[s_start:s_end, 5]
    
    # Resample to 512
    import scipy.signal
    ppg_512 = scipy.signal.resample(ppg, target_seq_len)
    acc_512 = scipy.signal.resample(acc, target_seq_len)
    
    # Normalize per segment (Zero mean, unit variance)
    ppg_norm = (ppg_512 - np.mean(ppg_512)) / (np.std(ppg_512) + 1e-6)
    acc_norm = (acc_512 - np.mean(acc_512)) / (np.std(acc_512) + 1e-6)
    
    return ppg_norm, acc_norm

ppg_walk, acc_walk = prepare_segments(rec_walk, 45)
ppg_run, acc_run = prepare_segments(rec_run, 45)

signals = {
    'PPG_Walk': ppg_walk,
    'ACC_Walk': acc_walk,
    'PPG_Run': ppg_run,
    'ACC_Run': acc_run
}

mask_ratios = [0.2, 0.4, 0.6]
eval_metrics = {}

plt.figure(figsize=(15, 10))
plot_idx = 1

for name, sig in signals.items():
    sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 512)
    eval_metrics[name] = {}
    
    for mask_ratio in mask_ratios:
        # Create random mask (1 = observed, 0 = masked)
        # MOMENT operates on patches of length 8
        num_patches = target_seq_len // 8 # 64 patches
        num_mask = int(num_patches * mask_ratio)
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        
        mask = torch.ones((1, target_seq_len), dtype=torch.float32)
        for idx in mask_indices:
            mask[:, idx*8 : (idx+1)*8] = 0.0
            
        with torch.no_grad():
            output = model(x_enc=sig_tensor, mask=mask)
            reconstruction = output.reconstruction.squeeze().cpu().numpy()
            
        orig = sig_tensor.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        
        # Calculate metrics on the MASKED positions
        masked_pos = (mask_np == 0)
        mae = float(np.mean(np.abs(orig[masked_pos] - reconstruction[masked_pos])))
        rmse = float(np.sqrt(np.mean((orig[masked_pos] - reconstruction[masked_pos])**2)))
        
        eval_metrics[name][f"mask_{int(mask_ratio*100)}%"] = {
            "mae": mae,
            "rmse": rmse
        }
        
    # Plot reconstruction for 40% mask
    plt.subplot(2, 2, plot_idx)
    plt.plot(orig, label="Ground Truth", color="black", lw=1.2)
    plt.plot(reconstruction, label="MOMENT Reconstruction", color="#e41a1c", lw=1.2, linestyle="--")
    # Highlight masked intervals
    for idx in mask_indices:
        plt.axvspan(idx*8, (idx+1)*8, color='yellow', alpha=0.25)
    plt.title(f"{name} (40% Masked Reconstruction) | MAE: {eval_metrics[name]['mask_40%']['mae']:.3f}", fontsize=11, fontweight="bold")
    plt.xlabel("Timesteps (64 Hz, 8s total)")
    plt.ylabel("Normalized Signal")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plot_idx += 1

plt.tight_layout()
plt.savefig("results/figures/exp2_moment_reconstruction.png", dpi=300)
plt.close()

with open("results/metrics/exp2_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("Experiment 2 complete! Saved plot to results/figures/exp2_moment_reconstruction.png and metrics to results/metrics/exp2_metrics.json")
for name, res in eval_metrics.items():
    print(f"[{name}] MAE @ 20%: {res['mask_20%']['mae']:.3f} | @ 40%: {res['mask_40%']['mae']:.3f} | @ 60%: {res['mask_60%']['mae']:.3f}")
