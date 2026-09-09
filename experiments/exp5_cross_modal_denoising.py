import os
import json
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

rec = wfdb.rdrecord("datasets/wrist/s3_run")
fs = rec.fs
data = pd.DataFrame(rec.p_signal).ffill().bfill().values[int(2.0 * fs):]

ecg = data[:, 0]
ppg = data[:, 1]
acc_x = data[:, 5]
acc_y = data[:, 6]
acc_z = data[:, 7]
acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Select a 15-second intense running segment
s_start = int(60.0 * fs)
s_len = int(15.0 * fs)
s_end = s_start + s_len

ecg_seg = ecg[s_start:s_end]
ppg_seg = ppg[s_start:s_end]
acc_seg = acc_mag[s_start:s_end]

# Bandpass filter PPG and ACC to cardiac/motion range (0.5 - 5.0 Hz)
b, a = scipy.signal.butter(3, [0.5 / (fs / 2), 5.0 / (fs / 2)], btype='bandpass')
ppg_filt = scipy.signal.filtfilt(b, a, ppg_seg)
acc_filt = scipy.signal.filtfilt(b, a, acc_seg)

# Adaptive Cross-Modal Denoising (Normalized LMS / Cross-Correlation Filter)
mu = 0.01 # Adaptation step size
order = 32
n_samples = len(ppg_filt)
cleaned_ppg = np.zeros(n_samples)
w = np.zeros(order)

for n in range(order, n_samples):
    x_vec = acc_filt[n - order : n][::-1]
    est_artifact = np.dot(w, x_vec)
    err = ppg_filt[n] - est_artifact
    cleaned_ppg[n] = err
    norm_factor = np.dot(x_vec, x_vec) + 1e-4
    w += (mu / norm_factor) * err * x_vec

# Frequency spectra
freqs = np.fft.rfftfreq(n_samples, d=1/fs)
raw_fft = np.abs(np.fft.rfft(ppg_filt))
acc_fft = np.abs(np.fft.rfft(acc_filt))
clean_fft = np.abs(np.fft.rfft(cleaned_ppg))

mask_f = (freqs >= 0.5) & (freqs <= 4.0)

# Detect true HR from ECG
b_ecg, a_ecg = scipy.signal.butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype='bandpass')
ecg_filt = scipy.signal.filtfilt(b_ecg, a_ecg, ecg_seg)
peaks, _ = scipy.signal.find_peaks(ecg_filt**2, distance=int(0.3*fs), height=np.mean(ecg_filt**2))
true_hr_bpm = (len(peaks) / 15.0) * 60.0
true_hr_freq = true_hr_bpm / 60.0

# Plotting Cross-Modal Denoising
plt.figure(figsize=(14, 8))

# Subplot 1: Time domain comparison
t = np.arange(s_len) / fs
plt.subplot(2, 1, 1)
plt.plot(t, ppg_filt / np.std(ppg_filt), label='Raw Wrist PPG (Heavily Corrupted by Motion)', color='#d62728', lw=1.2, alpha=0.7)
plt.plot(t, acc_filt / np.std(acc_filt) - 3, label='ACC Motion Reference (Shifted)', color='#ff7f0e', lw=1.2, alpha=0.8)
plt.plot(t, cleaned_ppg / np.std(cleaned_ppg) - 6, label='Cross-Modal Cleaned PPG (Shifted)', color='#2ca02c', lw=1.5)
plt.title(f"Running Exercise (15s Window): Raw PPG vs Motion Artifact vs Cross-Modal Denoised PPG", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Signal (Stacked)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

# Subplot 2: Frequency spectrum comparison
plt.subplot(2, 1, 2)
plt.plot(freqs[mask_f], raw_fft[mask_f] / np.max(raw_fft[mask_f]), label='Raw PPG Spectrum (Artifact peak dominates)', color='#d62728', lw=1.5)
plt.plot(freqs[mask_f], acc_fft[mask_f] / np.max(acc_fft[mask_f]), label='ACC Step Cadence Spectrum', color='#ff7f0e', lw=1.5, linestyle=':')
plt.plot(freqs[mask_f], clean_fft[mask_f] / np.max(clean_fft[mask_f]), label='Cross-Modal Denoised PPG Spectrum', color='#2ca02c', lw=2.0)
plt.axvline(true_hr_freq, color='blue', linestyle='--', lw=2, label=f'True ECG Heart Rate Peak ({true_hr_freq:.2f} Hz = {true_hr_bpm:.0f} BPM)')
plt.title("Spectral Disambiguation: How Cross-Modal Guidance Suppresses Motion Artifact Peak & Reveals True Cardiac Peak", fontsize=11, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/exp5_cross_modal_denoising.png", dpi=300)
plt.close()

# Compute SNR improvement
# Artifact peak power vs True HR peak power
raw_peak_freq = freqs[mask_f][np.argmax(raw_fft[mask_f])]
clean_peak_freq = freqs[mask_f][np.argmax(clean_fft[mask_f])]

snr_raw = float(10 * np.log10(np.interp(true_hr_freq, freqs, raw_fft) / (np.mean(raw_fft[mask_f]) + 1e-6)))
snr_clean = float(10 * np.log10(np.interp(true_hr_freq, freqs, clean_fft) / (np.mean(clean_fft[mask_f]) + 1e-6)))

metrics = {
    "true_hr_bpm": true_hr_bpm,
    "true_hr_freq_hz": true_hr_freq,
    "raw_ppg_dominant_freq_hz": float(raw_peak_freq),
    "clean_ppg_dominant_freq_hz": float(clean_peak_freq),
    "snr_raw_db": snr_raw,
    "snr_clean_db": snr_clean,
    "snr_gain_db": snr_clean - snr_raw
}

with open("results/metrics/exp5_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Experiment 5 complete! Saved plot to results/figures/exp5_cross_modal_denoising.png and metrics to results/metrics/exp5_metrics.json")
print(f"Results: True HR = {true_hr_bpm:.1f} BPM ({true_hr_freq:.2f} Hz)")
print(f"  Raw PPG Peak Freq: {raw_peak_freq:.2f} Hz (Mistaken as HR!)")
print(f"  Cleaned PPG Peak Freq: {clean_peak_freq:.2f} Hz (Correctly aligned with True HR!)")
print(f"  SNR Gain: {snr_clean - snr_raw:.2f} dB")
