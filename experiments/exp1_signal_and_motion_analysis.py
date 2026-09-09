import wfdb
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import json
import os

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

def compute_ecg_hr(ecg_signal, fs=256):
    # Bandpass filter 5-25 Hz (Pan-Tompkins stage)
    b, a = signal.butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, ecg_signal)
    diff = np.diff(filtered)
    squared = diff ** 2
    # Moving average 150ms
    window = int(0.15 * fs)
    kernel = np.ones(window) / window
    mwa = np.convolve(squared, kernel, mode='same')
    
    # Peak detection (min distance ~0.3s = 200 BPM max)
    min_dist = int(0.3 * fs)
    threshold = np.mean(mwa) + 0.5 * np.std(mwa)
    peaks, _ = signal.find_peaks(mwa, distance=min_dist, height=threshold)
    
    # Calculate HR in 8-second rolling windows (1-second step)
    duration_s = int(len(ecg_signal) / fs)
    window_s = 8
    time_points = []
    hr_values = []
    
    for t in range(0, duration_s - window_s, 1):
        t_start = t * fs
        t_end = (t + window_s) * fs
        win_peaks = peaks[(peaks >= t_start) & (peaks < t_end)]
        if len(win_peaks) >= 2:
            rr_intervals = np.diff(win_peaks) / fs
            mean_rr = np.median(rr_intervals)
            hr = 60.0 / mean_rr
            time_points.append(t + window_s / 2.0)
            hr_values.append(hr)
            
    return np.array(time_points), np.array(hr_values), peaks

def compute_acc_cadence(acc_mag, fs=256):
    # Bandpass filter 0.5 - 4.0 Hz (walking/running step frequency range: 30 - 240 steps/min)
    b, a = signal.butter(3, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, acc_mag)
    
    duration_s = int(len(acc_mag) / fs)
    window_s = 8
    time_points = []
    cadence_spm = []
    
    for t in range(0, duration_s - window_s, 1):
        t_start = t * fs
        t_end = (t + window_s) * fs
        win_sig = filtered[t_start:t_end]
        # FFT to find dominant step frequency
        freqs = np.fft.rfftfreq(len(win_sig), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(win_sig))
        # Mask between 0.8 and 3.5 Hz
        mask = (freqs >= 0.8) & (freqs <= 3.5)
        if np.any(mask) and np.max(fft_vals[mask]) > 0:
            dom_freq = freqs[mask][np.argmax(fft_vals[mask])]
            spm = dom_freq * 60.0
            time_points.append(t + window_s / 2.0)
            cadence_spm.append(spm)
            
    return np.array(time_points), np.array(cadence_spm)

records = {
    'walk': 'datasets/wrist/s1_walk',
    'run': 'datasets/wrist/s3_run',
    'bike': 'datasets/wrist/s1_low_resistance_bike'
}

metrics_summary = {}

plt.figure(figsize=(16, 12))

for i, (act, rec_path) in enumerate(records.items()):
    rec = wfdb.rdrecord(rec_path)
    fs = rec.fs
    # Use first 3 minutes for detailed analysis
    max_samples = 3 * 60 * fs
    ecg = rec.p_signal[:max_samples, 0]
    ppg = rec.p_signal[:max_samples, 1]
    acc_x = rec.p_signal[:max_samples, 5]
    acc_y = rec.p_signal[:max_samples, 6]
    acc_z = rec.p_signal[:max_samples, 7]
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    t_hr, hr_vals, peaks = compute_ecg_hr(ecg, fs)
    t_cad, cad_vals = compute_acc_cadence(acc_mag, fs)
    
    # Frequency analysis of 10-second segment
    seg_len = 10 * fs
    ppg_seg = ppg[10*fs:20*fs]
    acc_seg = acc_mag[10*fs:20*fs]
    
    # Detrend
    ppg_seg = ppg_seg - np.mean(ppg_seg)
    acc_seg = acc_seg - np.mean(acc_seg)
    
    freqs = np.fft.rfftfreq(seg_len, d=1/fs)
    ppg_fft = np.abs(np.fft.rfft(ppg_seg))
    acc_fft = np.abs(np.fft.rfft(acc_seg))
    
    # Normalize FFT
    ppg_fft /= (np.max(ppg_fft) + 1e-6)
    acc_fft /= (np.max(acc_fft) + 1e-6)
    
    # Store metrics
    metrics_summary[act] = {
        'mean_hr_bpm': float(np.mean(hr_vals)),
        'std_hr_bpm': float(np.std(hr_vals)),
        'mean_cadence_spm': float(np.mean(cad_vals)) if len(cad_vals) > 0 else 0.0,
        'mean_acc_energy': float(np.var(acc_mag))
    }
    
    # Plot 1: Waveforms (3 seconds)
    plt.subplot(3, 2, i * 2 + 1)
    t_plot = np.arange(0, 3 * fs) / fs
    plt.plot(t_plot, ppg[:3*fs], label='Wrist PPG', color='#1f77b4', lw=1.5)
    plt.plot(t_plot, acc_mag[:3*fs] - np.mean(acc_mag[:3*fs]), label='ACC Magnitude (Centered)', color='#ff7f0e', alpha=0.7, lw=1.2)
    plt.title(f"Activity: {act.upper()} - Waveforms (3s) | Mean HR: {np.mean(hr_vals):.1f} BPM", fontsize=11, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Signal")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Spectral Collision (PPG vs ACC)
    plt.subplot(3, 2, i * 2 + 2)
    mask_freq = (freqs >= 0.5) & (freqs <= 4.5)
    plt.plot(freqs[mask_freq], ppg_fft[mask_freq], label='PPG Spectrum', color='#1f77b4', lw=1.8)
    plt.plot(freqs[mask_freq], acc_fft[mask_freq], label='ACC Spectrum (Motion Artifact)', color='#d62728', lw=1.5, linestyle='--')
    true_hr_freq = np.mean(hr_vals) / 60.0
    plt.axvline(true_hr_freq, color='green', linestyle=':', lw=2, label=f'True HR Freq ({true_hr_freq:.2f} Hz = {np.mean(hr_vals):.0f} BPM)')
    plt.title(f"Spectral Collision Analysis: {act.upper()}", fontsize=11, fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/exp1_signal_and_spectral_collision.png", dpi=300)
plt.close()

with open("results/metrics/exp1_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("Experiment 1 complete! Saved plot to results/figures/exp1_signal_and_spectral_collision.png and metrics to results/metrics/exp1_metrics.json")
for k, v in metrics_summary.items():
    print(f"[{k.upper()}] Mean HR: {v['mean_hr_bpm']:.1f} BPM | Cadence: {v['mean_cadence_spm']:.1f} SPM | ACC Energy: {v['mean_acc_energy']:.3f}")
