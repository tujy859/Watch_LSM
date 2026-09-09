import sys
import os
import json
import torch
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.append(os.path.abspath("repos/limu_bert"))
from models import Transformer
from config import PretrainModelConfig

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# Load config
cfg = PretrainModelConfig(
    feature_num=6,
    hidden=72,
    hidden_ff=144,
    n_layers=4,
    n_heads=4,
    seq_len=120,
    emb_norm=True
)

model = Transformer(cfg)
weights_path = "repos/limu_bert/saved/pretrain_base_uci_20_120/uci.pt"

print(f"Loading pretrained LIMU-BERT weights from {weights_path}...")
state_dict = torch.load(weights_path, map_location="cpu")
cleaned_dict = {}
for k, v in state_dict.items():
    cleaned_key = k.replace("transformer.", "")
    if cleaned_key in model.state_dict():
        cleaned_dict[cleaned_key] = v

model.load_state_dict(cleaned_dict, strict=False)
model.eval()
print("LIMU-BERT loaded successfully!")

import wfdb

records = [
    ("walk", "datasets/wrist/s1_walk", 0),
    ("walk", "datasets/wrist/s2_walk", 0),
    ("run", "datasets/wrist/s3_run", 1),
    ("run", "datasets/wrist/s5_run", 1),
    ("bike", "datasets/wrist/s1_low_resistance_bike", 2)
]

all_embeddings = []
all_labels = []
all_cadences = []

target_seq_len = 120 # 6 seconds at 20 Hz

for label_name, path, label_id in records:
    rec = wfdb.rdrecord(path)
    fs = rec.fs # 256 Hz
    data = rec.p_signal # (N, 15)
    
    # Clean NaNs by pandas ffill/bfill
    df_data = pd.DataFrame(data).ffill().bfill().values
    
    # Skip first 2 seconds of sensor warm-up
    warmup = int(2.0 * fs)
    df_data = df_data[warmup:]
    
    # Extract 6-channel IMU: channels 5,6,7 (accel) + channels 2,3,4 (gyro)
    imu = np.hstack([df_data[:, 5:8], df_data[:, 2:5]])
    
    # 6-second sliding windows with 3-second step
    win_len = int(6.0 * fs)
    win_step = int(3.0 * fs)
    
    for s_start in range(0, len(imu) - win_len, win_step):
        chunk = imu[s_start : s_start + win_len]
        # Resample to 120 points (20 Hz)
        chunk_resampled = scipy.signal.resample(chunk, target_seq_len, axis=0)
        
        # Normalize per window
        std_val = np.std(chunk_resampled, axis=0)
        std_val[std_val < 1e-4] = 1.0
        chunk_norm = (chunk_resampled - np.mean(chunk_resampled, axis=0)) / std_val
        
        # Estimate cadence ground truth from ACC magnitude FFT
        acc_mag = np.sqrt(chunk[:, 0]**2 + chunk[:, 1]**2 + chunk[:, 2]**2)
        fft_vals = np.abs(np.fft.rfft(acc_mag - np.mean(acc_mag)))
        freqs = np.fft.rfftfreq(len(acc_mag), d=1/fs)
        mask = (freqs >= 0.8) & (freqs <= 3.5)
        cadence = freqs[mask][np.argmax(fft_vals[mask])] * 60.0 if np.any(mask) else 0.0
        
        # Forward pass through LIMU-BERT
        inp_tensor = torch.tensor(chunk_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model(inp_tensor) # (1, 120, 72)
            pooled_emb = emb.mean(dim=1).squeeze().numpy()
            
        all_embeddings.append(pooled_emb)
        all_labels.append(label_id)
        all_cadences.append(cadence)

X = np.array(all_embeddings)
y = np.array(all_labels)
y_cad = np.array(all_cadences)

# Verify no NaNs
assert not np.isnan(X).any(), "X still contains NaN"

print(f"Extracted {len(X)} clean segments with embedding shape {X.shape}")

# Evaluate representation quality:
# 1. Silhouette Score of raw embeddings across activities
sil_score = float(silhouette_score(X, y))

# 2. Linear probe classification accuracy (Train/Test split 70/30)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

clf = LogisticRegression(max_iter=500)
clf.fit(X[train_idx], y[train_idx])
acc_score = float(clf.score(X[test_idx], y[test_idx]))

# 3. Ridge regression for Cadence (SPM) estimation from embeddings
reg = Ridge(alpha=1.0)
reg.fit(X[train_idx], y_cad[train_idx])
pred_cad = reg.predict(X[test_idx])
mae_cad = float(np.mean(np.abs(pred_cad - y_cad[test_idx])))
r2_cad = float(reg.score(X[test_idx], y_cad[test_idx]))

# 4. PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(13, 5))

# Plot 1: PCA Clusters
plt.subplot(1, 2, 1)
colors = ['#1f77b4', '#d62728', '#2ca02c']
labels_str = ['Walking', 'Running', 'Biking']
for l_id, name, col in zip([0, 1, 2], labels_str, colors):
    mask = (y == l_id)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, color=col, alpha=0.75, edgecolors='none', s=40)
plt.title(f"LIMU-BERT Latent Space (PCA)\nSilhouette Score: {sil_score:.3f} | Activity Linear Probe Acc: {acc_score*100:.1f}%", fontsize=11, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cadence Prediction Parity Plot
plt.subplot(1, 2, 2)
plt.scatter(y_cad[test_idx], pred_cad, color='#9467bd', alpha=0.7, edgecolors='black', s=45)
plt.plot([40, 200], [40, 200], 'k--', lw=1.8, label='Ideal 1:1 Parity Line')
plt.title(f"Cadence Estimation from Foundation Embeddings\nMAE: {mae_cad:.2f} SPM | R2: {r2_cad:.3f}", fontsize=11, fontweight='bold')
plt.xlabel("Ground Truth Cadence (SPM)")
plt.ylabel("Predicted Cadence (SPM)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/exp3_limu_bert_embeddings.png", dpi=300)
plt.close()

metrics = {
    "num_samples": len(X),
    "embedding_dim": int(X.shape[1]),
    "activity_silhouette_score": sil_score,
    "activity_classification_accuracy": acc_score,
    "cadence_mae_spm": mae_cad,
    "cadence_r2": r2_cad
}

with open("results/metrics/exp3_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Experiment 3 complete! Saved plot to results/figures/exp3_limu_bert_embeddings.png and metrics to results/metrics/exp3_metrics.json")
print(f"Results: Silhouette = {sil_score:.3f} | Classification Acc = {acc_score*100:.1f}% | Cadence MAE = {mae_cad:.2f} SPM | R2 = {r2_cad:.3f}")
