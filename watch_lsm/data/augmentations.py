import numpy as np

class PhysiologicalAugmenter:
    """
    Data augmentation suite specifically tailored for wearable PPG and ACC signals.
    """
    def __init__(self, p_noise=0.5, p_mask=0.5, p_drift=0.5, p_motion_inject=0.3):
        self.p_noise = p_noise
        self.p_mask = p_mask
        self.p_drift = p_drift
        self.p_motion_inject = p_motion_inject

    def add_gaussian_noise(self, signal, snr_range=(15, 30)):
        if np.random.rand() > self.p_noise:
            return signal
        snr_db = np.random.uniform(*snr_range)
        sig_power = np.mean(signal ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power + 1e-8), size=signal.shape)
        return signal + noise

    def add_baseline_drift(self, signal, fs=64):
        if np.random.rand() > self.p_drift:
            return signal
        t = np.arange(signal.shape[-1]) / fs
        freq = np.random.uniform(0.1, 0.3)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.2, 0.6) * np.std(signal)
        drift = amp * np.sin(2 * np.pi * freq * t + phase)
        return signal + drift

    def random_patch_cutout(self, signal, max_mask_ratio=0.25):
        if np.random.rand() > self.p_mask:
            return signal
        seq_len = signal.shape[-1]
        mask_len = int(seq_len * np.random.uniform(0.05, max_mask_ratio))
        start = np.random.randint(0, seq_len - mask_len)
        sig_cut = signal.copy()
        sig_cut[..., start : start + mask_len] = 0.0
        return sig_cut

    def inject_motion_artifact(self, ppg_signal, acc_signal):
        """
        Synthetically couples ACC motion energy into the optical PPG channel.
        ppg_signal: (1, L_ppg)
        acc_signal: (3, L_acc)
        """
        if np.random.rand() > self.p_motion_inject:
            return ppg_signal
            
        from scipy.signal import resample
        if acc_signal.shape[-1] != ppg_signal.shape[-1]:
            acc_resamp = resample(acc_signal, ppg_signal.shape[-1], axis=-1)
        else:
            acc_resamp = acc_signal
            
        # Compute scalar motion magnitude (1, L_ppg)
        acc_mag = np.sqrt(np.sum(acc_resamp**2, axis=0, keepdims=True))
        acc_norm = (acc_mag - np.mean(acc_mag)) / (np.std(acc_mag) + 1e-5)
        coupling_weight = np.random.uniform(0.2, 0.5) * (np.std(ppg_signal) + 1e-5)
        
        out = ppg_signal + coupling_weight * acc_norm
        return out.astype(np.float32)

    def __call__(self, ppg, acc, fs=64):
        ppg_aug = self.add_gaussian_noise(ppg)
        ppg_aug = self.add_baseline_drift(ppg_aug, fs=fs)
        ppg_aug = self.random_patch_cutout(ppg_aug)
        ppg_aug = self.inject_motion_artifact(ppg_aug, acc)
        
        acc_aug = self.add_gaussian_noise(acc)
        acc_aug = self.random_patch_cutout(acc_aug)
        return ppg_aug.astype(np.float32), acc_aug.astype(np.float32)
