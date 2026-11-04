"""Fixed, label-free PPG-only heart-rate baselines for the evaluation PoC.

The Elgendi-style detector is an independent implementation of Algorithm IV in
Elgendi et al. (2013), doi:10.1371/journal.pone.0076585 (CC BY article). No external
implementation is copied. Window-local filtering, boundary handling, and median
interbeat-interval HR reduction are explicit PoC choices, not a claim of exact
reproduction of the paper's continuous-record peak-detection experiment.

Both algorithms are offline, process each window independently, and use no
reference HR, participant identity, activity, or neighbouring windows. A finite
estimate is not a signal-quality guarantee. Evaluation must retain failures and
report coverage rather than silently remove failed windows.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch


BASELINE_PROTOCOL = {
    "version": 1,
    "input": "Raw-amplitude single-channel PPG windows; positive systolic peaks",
    "output": "Heart rate in bpm; failures are NaN; no label-dependent tuning",
    "shared": {
        "mode": "Offline, independent windows; no temporal tracking or ACC",
        "normalization": "Per-window max-absolute scaling, centering, standard deviation",
        "minimum_window_seconds": 2.0,
        "invalid_input": "Any nonfinite sample or constant window yields NaN",
        "filter": "scipy Butterworth N=2, SOS forward-backward, default odd padding",
    },
    "welch": {
        "filter_hz": [0.5, 4.0],
        "search_hz": [0.5, 3.5],
        "segment_seconds": 4.0,
        "overlap": 0.5,
        "window": "hann",
        "detrend": "constant",
        "nfft": "Next power of two >= 8 * segment samples (zero-padding only)",
        "estimate": "Dominant PSD frequency, three-bin log-PSD parabolic interpolation",
        "source": "https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.signal.welch.html",
    },
    "elgendi": {
        "implementation": "Independent paper-informed Elgendi-style detector",
        "filter_hz": [0.5, 8.0],
        "peak_window_seconds": 0.111,
        "beat_window_seconds": 0.667,
        "window_rounding": "Nearest positive odd sample count; ties round up",
        "offset": 0.02,
        "moving_average_boundary": "Zero padding",
        "block_acceptance": "Short MA > long MA + offset * mean squared positive signal; width >= short MA",
        "peak_location": "Maximum positive filtered sample within each accepted block",
        "estimate": "60 / median interpeak interval in seconds; at least two peaks",
        "interval_rejection": "None; no HR clipping, refractory period, or outlier removal",
        "source": "https://doi.org/10.1371/journal.pone.0076585",
    },
}

_BATCH_SIZE = 512


def _nearest_odd(samples: float) -> int:
    return max(1, 2 * int(np.floor(samples / 2)) + 1)


def _welch_hr(filtered: np.ndarray, fs: float) -> np.ndarray:
    segment = min(filtered.shape[1], int(round(4.0 * fs)))
    nfft = 1 << (8 * segment - 1).bit_length()
    frequencies, power = welch(
        filtered,
        fs=fs,
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        nfft=nfft,
        detrend="constant",
        axis=-1,
    )
    candidates = np.flatnonzero((frequencies >= 0.5) & (frequencies <= 3.5))
    index = candidates[np.argmax(power[:, candidates], axis=1)]
    rows = np.arange(len(filtered))
    log_power = np.log(np.maximum(power, np.finfo(np.float64).tiny))
    left = log_power[rows, index - 1]
    center = log_power[rows, index]
    right = log_power[rows, index + 1]
    curvature = left - 2.0 * center + right
    offset = np.zeros(len(filtered), dtype=np.float64)
    interior = (index > candidates[0]) & (index < candidates[-1]) & (curvature < 0)
    offset[interior] = 0.5 * (left[interior] - right[interior]) / curvature[interior]
    estimate = (frequencies[index] + np.clip(offset, -0.5, 0.5) * fs / nfft) * 60.0
    estimate[power[rows, index] <= 0] = np.nan
    return estimate


def _elgendi_hr(filtered: np.ndarray, fs: float) -> np.ndarray:
    short = _nearest_odd(0.111 * fs)
    long = _nearest_odd(0.667 * fs)
    squared = np.maximum(filtered, 0.0) ** 2
    peak_average = uniform_filter1d(squared, short, axis=-1, mode="constant")
    beat_average = uniform_filter1d(squared, long, axis=-1, mode="constant")
    threshold = beat_average + 0.02 * squared.mean(axis=1, keepdims=True)
    blocks = peak_average > threshold
    result = np.full(len(filtered), np.nan, dtype=np.float64)
    # Filtering and moving averages are vectorized; only the few peak blocks in
    # each short window require a loop. Padding also handles edge-touching blocks.
    for row, mask in enumerate(blocks):
        changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        peaks = [
            start + int(np.argmax(filtered[row, start:stop]))
            for start, stop in zip(starts, stops)
            if stop - start >= short
        ]
        if len(peaks) >= 2:
            result[row] = 60.0 * fs / np.median(np.diff(peaks))
    return result


def predict_hr_windows(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Return ``welch`` and ``elgendi`` HR arrays for ``x`` shaped [N, T].

    ``fs`` must exceed 16 Hz to support the published 8 Hz filter. Windows under
    two seconds, flatlines and windows containing NaN/inf return NaN. Shape and
    sampling-rate errors raise ValueError. Inputs are never modified. Processing
    is batched to bound scratch memory for large datasets and numpy memmaps.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape [n_windows, n_samples]")
    if not np.isfinite(fs) or fs <= 16.0:
        raise ValueError("fs must be finite and greater than 16 Hz")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("x must contain real numeric samples")
    result = {name: np.full(len(x), np.nan, dtype=np.float64) for name in ("welch", "elgendi")}
    if x.shape[1] < int(np.ceil(2.0 * fs)) or len(x) == 0:
        return result

    welch_filter = butter(2, [0.5, 4.0], btype="bandpass", fs=fs, output="sos")
    peak_filter = butter(2, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    for start in range(0, len(x), _BATCH_SIZE):
        batch = np.asarray(x[start:start + _BATCH_SIZE], dtype=np.float64)
        valid = np.isfinite(batch).all(axis=1)
        positions = np.flatnonzero(valid)
        if not len(positions):
            continue
        values = batch[positions].copy()
        scale = np.max(np.abs(values), axis=1, keepdims=True)
        # Initial scaling avoids overflow and preserves arbitrarily small
        # nonconstant amplitudes. Every operation remains local to one window.
        values /= np.where(scale > 0, scale, 1.0)
        values -= values.mean(axis=1, keepdims=True)
        deviation = values.std(axis=1)
        nonconstant = deviation > 0
        positions = positions[nonconstant] + start
        values = values[nonconstant]
        if not len(values):
            continue
        values /= deviation[nonconstant, None]
        result["welch"][positions] = _welch_hr(sosfiltfilt(welch_filter, values, axis=-1), fs)
        result["elgendi"][positions] = _elgendi_hr(sosfiltfilt(peak_filter, values, axis=-1), fs)
    return result
