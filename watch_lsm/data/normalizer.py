import numpy as np

def robust_zscore(signal, axis=-1, clip_std=4.0):
    """
    Robust Z-Score Normalization with median and IQR/MAD, or standard mean/std with clipping.
    """
    mean = np.mean(signal, axis=axis, keepdims=True)
    std = np.std(signal, axis=axis, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    norm = (signal - mean) / std
    if clip_std is not None:
        norm = np.clip(norm, -clip_std, clip_std)
    return norm

def compute_ppg_sqi(ppg_segment, fs=64):
    """
    Computes Signal Quality Index (SQI) for a PPG segment.
    Returns:
        quality_score: float in [0.0, 1.0]
        is_usable: bool
    """
    if len(ppg_segment) == 0:
        return 0.0, False
        
    # Check for flatlines or constant values
    std_val = np.std(ppg_segment)
    if std_val < 1e-4:
        return 0.0, False
        
    # Perfusion Index proxy (relative amplitude)
    ptp = np.ptp(ppg_segment)
    
    # Skewness and Kurtosis
    from scipy.stats import skew, kurtosis
    sk = abs(skew(ppg_segment))
    kt = kurtosis(ppg_segment)
    
    # Clean pulsatile PPG typically has moderate skewness and low baseline drift
    score = 1.0
    if sk > 2.5:
        score -= 0.3
    if kt > 6.0:
        score -= 0.3
    if std_val > 10.0: # Huge motion saturated spike
        score -= 0.4
        
    score = max(0.0, min(1.0, score))
    return score, score >= 0.5
