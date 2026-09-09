import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# Define PaPaGei 1D-ResNet architecture
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class PPG_Foundation_Encoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, emb_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResNet1DBlock(base_filters, base_filters * 2, stride=2)
        self.layer2 = ResNet1DBlock(base_filters * 2, base_filters * 4, stride=2)
        self.layer3 = ResNet1DBlock(base_filters * 4, emb_dim, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        emb = self.gap(h).squeeze(-1) # (B, emb_dim)
        return emb

ppg_encoder = PPG_Foundation_Encoder(in_channels=1, base_filters=32, emb_dim=256)
acc_encoder = PPG_Foundation_Encoder(in_channels=3, base_filters=32, emb_dim=256)

ppg_encoder.eval()
acc_encoder.eval()

# Load datasets and prepare 8-second segments (target fs = 125 Hz, length 1000)
import wfdb

target_fs = 125
seg_samples = 8 * target_fs # 1000 samples

records = [
    ("walk", "datasets/wrist/s1_walk"),
    ("walk", "datasets/wrist/s2_walk"),
    ("run", "datasets/wrist/s3_run"),
    ("run", "datasets/wrist/s5_run"),
    ("bike", "datasets/wrist/s1_low_resistance_bike")
]

ppg_embeddings = []
acc_embeddings = []
heuristic_hrs = []
true_hrs = []
activities = []

for act, rec_path in records:
    rec = wfdb.rdrecord(rec_path)
    fs = rec.fs # 256 Hz
    data = pd.DataFrame(rec.p_signal).ffill().bfill().values
    
    # Warmup
    data = data[int(2.0 * fs):]
    
    ecg = data[:, 0]
    ppg = data[:, 1]
    acc = data[:, 5:8] # 3-axis low noise accel
    
    # ECG R-peak ground truth detection
    b, a = scipy.signal.butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype='bandpass')
    ecg_filt = scipy.signal.filtfilt(b, a, ecg)
    ecg_diff = np.diff(ecg_filt) ** 2
    win = int(0.15 * fs)
    ecg_mwa = np.convolve(ecg_diff, np.ones(win)/win, mode='same')
    peaks, _ = scipy.signal.find_peaks(ecg_mwa, distance=int(0.3 * fs), height=np.mean(ecg_mwa))
    
    win_len_fs = int(8.0 * fs)
    win_step_fs = int(2.0 * fs)
    
    for start in range(0, len(data) - win_len_fs, win_step_fs):
        end = start + win_len_fs
        # Reference HR
        win_peaks = peaks[(peaks >= start) & (peaks < end)]
        if len(win_peaks) < 3:
            continue
        rr = np.diff(win_peaks) / fs
        hr_gt = 60.0 / np.median(rr)
        if hr_gt < 40 or hr_gt > 220:
            continue
            
        # PPG segment
        ppg_win = ppg[start:end]
        acc_win = acc[start:end]
        
        # Heuristic HR: dominant peak in PPG FFT (0.8 - 3.5 Hz)
        ppg_filt = scipy.signal.detrend(ppg_win)
        fft_vals = np.abs(np.fft.rfft(ppg_filt))
        freqs = np.fft.rfftfreq(len(ppg_filt), d=1/fs)
        mask = (freqs >= 0.8) & (freqs <= 3.5)
        heur_hr = freqs[mask][np.argmax(fft_vals[mask])] * 60.0 if np.any(mask) else 75.0
        
        # Resample to 125 Hz (1000 points)
        ppg_resampled = scipy.signal.resample(ppg_win, seg_samples)
        acc_resampled = scipy.signal.resample(acc_win, seg_samples, axis=0)
        
        # Normalize
        ppg_norm = (ppg_resampled - np.mean(ppg_resampled)) / (np.std(ppg_resampled) + 1e-5)
        acc_norm = (acc_resampled - np.mean(acc_resampled, axis=0)) / (np.std(acc_resampled, axis=0) + 1e-5)
        
        # Tensors
        t_ppg = torch.tensor(ppg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 1000)
        t_acc = torch.tensor(acc_norm.T, dtype=torch.float32).unsqueeze(0) # (1, 3, 1000)
        
        with torch.no_grad():
            e_ppg = ppg_encoder(t_ppg).squeeze(0).numpy()
            e_acc = acc_encoder(t_acc).squeeze(0).numpy()
            
        ppg_embeddings.append(e_ppg)
        acc_embeddings.append(e_acc)
        heuristic_hrs.append(heur_hr)
        true_hrs.append(hr_gt)
        activities.append(act)

E_ppg = np.array(ppg_embeddings)
E_acc = np.array(acc_embeddings)
E_multi = np.hstack([E_ppg, E_acc]) # (N, 512)
Y_true = np.array(true_hrs)
Y_heur = np.array(heuristic_hrs)
activities = np.array(activities)

print(f"Total evaluated segments: {len(Y_true)}")

# Train/test split stratified by sample index
np.random.seed(42)
perm = np.random.permutation(len(Y_true))
split = int(0.7 * len(Y_true))
train_idx, test_idx = perm[:split], perm[split:]

# 1. Classical Heuristic HR error
mae_heur = float(mean_absolute_error(Y_true[test_idx], Y_heur[test_idx]))

# 2. PPG Foundation Linear Probe
reg_ppg = Ridge(alpha=10.0)
reg_ppg.fit(E_ppg[train_idx], Y_true[train_idx])
pred_ppg = reg_ppg.predict(E_ppg[test_idx])
mae_ppg = float(mean_absolute_error(Y_true[test_idx], pred_ppg))
r2_ppg = float(r2_score(Y_true[test_idx], pred_ppg))

# 3. Multi-Modal (PPG + ACC) Foundation Linear Probe
reg_multi = Ridge(alpha=10.0)
reg_multi.fit(E_multi[train_idx], Y_true[train_idx])
pred_multi = reg_multi.predict(E_multi[test_idx])
mae_multi = float(mean_absolute_error(Y_true[test_idx], pred_multi))
r2_multi = float(r2_score(Y_true[test_idx], pred_multi))

# Breakdown by activity in test set
act_test = activities[test_idx]
breakdown = {}
for act in ['walk', 'run', 'bike']:
    m = (act_test == act)
    if np.any(m):
        breakdown[act] = {
            "num_samples": int(np.sum(m)),
            "mae_heuristic": float(mean_absolute_error(Y_true[test_idx][m], Y_heur[test_idx][m])),
            "mae_ppg_fm": float(mean_absolute_error(Y_true[test_idx][m], pred_ppg[m])),
            "mae_multimodal_fm": float(mean_absolute_error(Y_true[test_idx][m], pred_multi[m]))
        }

plt.figure(figsize=(14, 6))

# Plot 1: Overall Parity Comparison
plt.subplot(1, 2, 1)
plt.scatter(Y_true[test_idx], Y_heur[test_idx], color='gray', alpha=0.4, label=f'Classical Heuristic (MAE={mae_heur:.2f} BPM)', s=30)
plt.scatter(Y_true[test_idx], pred_ppg, color='#1f77b4', alpha=0.6, label=f'PPG-FM Alone (MAE={mae_ppg:.2f} BPM)', s=35)
plt.scatter(Y_true[test_idx], pred_multi, color='#2ca02c', alpha=0.7, edgecolors='black', label=f'Multi-Modal FM (PPG+ACC) (MAE={mae_multi:.2f} BPM)', s=45)
plt.plot([40, 180], [40, 180], 'k--', lw=2, label='ECG Reference (Ground Truth)')
plt.title("Heart Rate Estimation: Heuristic vs PPG-FM vs Multi-Modal FM", fontsize=11, fontweight='bold')
plt.xlabel("ECG Reference Ground Truth HR (BPM)")
plt.ylabel("Estimated HR (BPM)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Activity Breakdown Bar Chart
plt.subplot(1, 2, 2)
acts = list(breakdown.keys())
x = np.arange(len(acts))
width = 0.25
plt.bar(x - width, [breakdown[a]['mae_heuristic'] for a in acts], width, label='Heuristic Peak', color='#7f7f7f', alpha=0.85)
plt.bar(x, [breakdown[a]['mae_ppg_fm'] for a in acts], width, label='PPG-FM', color='#1f77b4', alpha=0.85)
plt.bar(x + width, [breakdown[a]['mae_multimodal_fm'] for a in acts], width, label='Multi-Modal (PPG+ACC)', color='#2ca02c', alpha=0.85)
plt.xticks(x, [a.upper() for a in acts], fontweight='bold')
plt.title("HR Estimation MAE (BPM) across Activity States", fontsize=11, fontweight='bold')
plt.ylabel("Mean Absolute Error (BPM, lower is better)")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/figures/exp4_heart_rate_benchmarking.png", dpi=300)
plt.close()

metrics = {
    "overall": {
        "mae_heuristic": mae_heur,
        "mae_ppg_fm": mae_ppg,
        "r2_ppg_fm": r2_ppg,
        "mae_multimodal_fm": mae_multi,
        "r2_multimodal_fm": r2_multi,
        "error_reduction_percent": float((mae_heur - mae_multi) / mae_heur * 100)
    },
    "by_activity": breakdown
}

with open("results/metrics/exp4_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Experiment 4 complete! Saved plot to results/figures/exp4_heart_rate_benchmarking.png and metrics to results/metrics/exp4_metrics.json")
print(f"Overall HR MAE -> Heuristic: {mae_heur:.2f} BPM | PPG-FM: {mae_ppg:.2f} BPM | MultiModal (PPG+ACC): {mae_multi:.2f} BPM")
for a, d in breakdown.items():
    print(f"  [{a.upper()}] Heuristic: {d['mae_heuristic']:.2f} | PPG-FM: {d['mae_ppg_fm']:.2f} | MultiModal: {d['mae_multimodal_fm']:.2f} BPM")
