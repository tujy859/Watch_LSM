import sys
import os
import time
import json
import torch
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import wfdb

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

print("Initializing Mantis-8M on CPU...")
t0 = time.time()
network = Mantis8M(device="cpu")
network = network.from_pretrained("paris-noah/Mantis-8M")
trainer = MantisTrainer(device="cpu", network=network)
print(f"Mantis-8M loaded in {time.time() - t0:.2f}s!")

# Load smartwatch datasets
records = [
    ("walk", "datasets/wrist/s1_walk", 0),
    ("walk", "datasets/wrist/s2_walk", 0),
    ("run", "datasets/wrist/s3_run", 1),
    ("run", "datasets/wrist/s5_run", 1),
    ("bike", "datasets/wrist/s1_low_resistance_bike", 2)
]

raw_segments = []
labels = []
target_len = 512 # Mantis default input length

for act_name, path, act_id in records:
    rec = wfdb.rdrecord(path)
    fs = rec.fs # 256 Hz
    df = pd.DataFrame(rec.p_signal).ffill().bfill().values
    df = df[int(2.0 * fs):] # Skip warmup
    
    # 3-axis ACC: channels 5, 6, 7
    acc = df[:, 5:8]
    
    # 8-second window (2048 points), step 2-second (512 points)
    win_len = int(8.0 * fs)
    win_step = int(2.0 * fs)
    
    for s_start in range(0, len(acc) - win_len, win_step):
        chunk = acc[s_start : s_start + win_len]
        # Resample to 512 points
        chunk_resamp = scipy.signal.resample(chunk, target_len, axis=0) # (512, 3)
        # Normalize per window
        std_val = np.std(chunk_resamp, axis=0)
        std_val[std_val < 1e-4] = 1.0
        chunk_norm = (chunk_resamp - np.mean(chunk_resamp, axis=0)) / std_val
        
        raw_segments.append(chunk_norm.T) # (3, 512)
        labels.append(act_id)

X_all = np.array(raw_segments, dtype=np.float32) # (N, 3, 512)
y_all = np.array(labels)
print(f"Prepared {len(X_all)} smartwatch ACC segments of shape {X_all.shape}")

# Extract features with Mantis
print("Extracting discriminative representations with Mantis...")
t_start = time.time()
Z_mantis = trainer.transform(X_all) # (N, 768)
dt_extract = time.time() - t_start
ms_per_sample = (dt_extract / len(X_all)) * 1000
print(f"Extracted {Z_mantis.shape} features in {dt_extract:.2f}s ({ms_per_sample:.2f} ms/sample on CPU)")

# Evaluate Representation Quality
sil_score = float(silhouette_score(Z_mantis, y_all))

# Train/Test Split (70/30)
np.random.seed(42)
perm = np.random.permutation(len(y_all))
split = int(0.7 * len(y_all))
train_idx, test_idx = perm[:split], perm[split:]

clf = LogisticRegression(max_iter=500)
clf.fit(Z_mantis[train_idx], y_all[train_idx])
acc_test = float(clf.score(Z_mantis[test_idx], y_all[test_idx]))
y_pred = clf.predict(Z_mantis[test_idx])

report = classification_report(y_all[test_idx], y_pred, target_names=["Walk", "Run", "Bike"], output_dict=True)

# PCA Visualization
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z_mantis)

plt.figure(figsize=(12, 5))

# Plot 1: PCA Scatter
plt.subplot(1, 2, 1)
colors = ['#1f77b4', '#d62728', '#2ca02c']
labels_str = ['Walking', 'Running', 'Biking']
for l_id, name, col in zip([0, 1, 2], labels_str, colors):
    m = (y_all == l_id)
    plt.scatter(Z_pca[m, 0], Z_pca[m, 1], label=name, color=col, alpha=0.75, edgecolors='none', s=40)
plt.title(f"Mantis Latent Space (PCA)\nSilhouette Score: {sil_score:.3f} | Activity Probe Acc: {acc_test*100:.1f}%", fontsize=11, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Per-Class F1 Score
plt.subplot(1, 2, 2)
classes = ["Walk", "Run", "Bike"]
f1_scores = [report[c]["f1-score"] for c in classes]
bars = plt.bar(classes, f1_scores, color=['#1f77b4', '#d62728', '#2ca02c'], width=0.5, alpha=0.85)
for bar, val in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')
plt.ylim(0, 1.15)
plt.title(f"Mantis Zero-Shot Linear Probe F1-Scores (CPU: {ms_per_sample:.1f}ms)", fontsize=11, fontweight='bold')
plt.ylabel("F1-Score")
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("results/figures/exp6_mantis_clustering.png", dpi=300)
plt.close()

# Plot 3: Standalone Confusion Matrix Heatmap
cm = confusion_matrix(y_all[test_idx], y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Mantis Activity Classification Confusion Matrix\n(Test Acc: {acc_test*100:.1f}%)", fontsize=11, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=10)
plt.yticks(tick_marks, classes, fontsize=10)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontweight='bold', fontsize=12)
plt.ylabel('True Label', fontsize=10)
plt.xlabel('Predicted Label', fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/exp6_mantis_confusion_matrix.png", dpi=300)
plt.close()

# Plot 4: Standalone High-Res PCA Scatter
plt.figure(figsize=(7, 6))
for l_id, name, col in zip([0, 1, 2], labels_str, colors):
    m = (y_all == l_id)
    plt.scatter(Z_pca[m, 0], Z_pca[m, 1], label=name, color=col, alpha=0.75, edgecolors='none', s=45)
plt.title(f"Mantis Latent Embeddings (PCA 2D Projection)\nSilhouette Score: {sil_score:.3f}", fontsize=12, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% explained variance)", fontsize=10)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% explained variance)", fontsize=10)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/exp6_mantis_pca_embeddings.png", dpi=300)
plt.close()

metrics = {
    "num_samples": len(X_all),
    "feature_dim": int(Z_mantis.shape[1]),
    "cpu_latency_ms_per_sample": ms_per_sample,
    "silhouette_score": sil_score,
    "classification_accuracy": acc_test,
    "per_class_f1": {c: report[c]["f1-score"] for c in classes},
    "confusion_matrix": cm.tolist()
}

with open("results/metrics/exp6_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Experiment 6 complete! Saved plots to results/figures/ and metrics to results/metrics/exp6_metrics.json")
print(f"Results: Test Accuracy = {acc_test*100:.1f}% | Silhouette = {sil_score:.3f} | Latency = {ms_per_sample:.2f} ms/sample on CPU")
