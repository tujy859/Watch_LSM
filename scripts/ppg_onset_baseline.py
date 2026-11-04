# SPDX-License-Identifier: MIT
"""Fixed MSPTDfast-style pulse onset detector for the manual wrist PPG PoC.

Adapted from the MIT-licensed MSPTDfast v2 implementation by Peter H. Charlton
in ppg-beats v2.3, commit c397be7afc0d8a1601a85cd673718372e40f1f59:
https://github.com/peterhcharlton/ppg-beats/blob/c397be7afc0d8a1601a85cd673718372e40f1f59/source/msptdfastv2_beat_detector.m
Algorithm paper: Charlton et al. (2025), doi:10.1088/1361-6579/adb89e.

This compact Python adaptation retains six-second windows, 20% overlap,
integer downsampling to >=20 Hz, linearly detrended binary scalograms limited
to >=30 bpm, and local 50 ms refinement. It adds a fixed third-order 0.5-8 Hz
zero-phase bandpass for raw input, correct zero-based sample mapping, bounded
refinement, and explicit polarity. It is not claimed to be bitwise equivalent
to the Matlab implementation, nor a reproduction of its full benchmark.

EMBC21/Biagetti raw reflected-light intensity has the opposite sign to blood
volume. Its expert diastolic-onset annotations therefore correspond to raw
MAXIMA. For this dataset use the default ``polarity='raw_intensity'``. For an
explicitly inverted blood-volume waveform use ``polarity='blood_volume'``;
the latter finds minima. Polarity is an acquisition convention, never selected
from test annotations. Source: Wolling et al. (2021), EMBC, pp. 1157-1163,
https://www.eti.uni-siegen.de/ubicomp/papers/ubi_embc2021.pdf .

MIT License
Copyright (c) 2022, 2024 Peter H. Charlton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

ONSET_BASELINE_PROTOCOL = {
    "name": "MSPTDfast-style",
    "version": 1,
    "source_commit": "c397be7afc0d8a1601a85cd673718372e40f1f59",
    "source_license": "MIT, Copyright (c) 2022, 2024 Peter H. Charlton",
    "paper": "https://doi.org/10.1088/1361-6579/adb89e",
    "input": "continuous real PPG; no ECG, labels, temporal realignment or ACC",
    "polarity": "EMBC21 raw intensity maxima = diastolic onset; fixed from acquisition physics",
    "filter": "third-order Butterworth 0.5-8 Hz, SOS forward-backward over the supplied continuous segment",
    "window_seconds": 6., "window_overlap": .2,
    "downsampling": "greatest integer factor leaving fs >=20 Hz after fixed bandpass",
    "scalogram": "linear detrend; strict maxima; scales corresponding to >=30 bpm; first maximally populated scale",
    "refinement": "maximum of original-rate filtered input within +/-50ms; clip search to window",
    "duplicates": "retain sorted unique original sample indices; no additional refractory or interval rejection",
    "boundaries": "window edges lack full multiscale support; supply real continuous context and score fixed common core",
    "timing": "no fitted lag; returned event time is original zero-based sample index divided by fs",
    "limitation": "Python adaptation with fixed raw-input filter, not bitwise-verified Matlab reproduction; filter can shift morphological fiducials",
}


def _window_starts(length, window, step):
    """Cover the full acquisition, appending one tail-aligned window if needed."""
    if length <= window:
        return np.array([0], dtype=int)
    starts = np.arange(0, length - window + 1, step, dtype=int)
    if starts[-1] + window < length:
        starts = np.r_[starts, length - window]
    return starts


def _multiscale_maxima(signal, fs):
    """MSPTD binary local maxima intersection at the most populated scale."""
    x = detrend(np.asarray(signal, dtype=float), type="linear")
    n = len(x)
    largest = int(np.ceil(n / 2.)) - 1
    if largest < 1:
        return np.array([], dtype=int)
    # The reference uses (L / scale) / window_duration as its scale frequency.
    scales = np.arange(1, largest + 1)
    allowed = (largest / scales) / (n / fs) >= .5
    if not allowed.any():
        return np.array([], dtype=int)
    maximum = int(scales[allowed][-1])
    local = np.zeros((maximum, n), dtype=bool)
    for scale in range(1, maximum + 1):
        local[scale - 1, scale:n - scale] = (x[scale:n - scale] > x[:n - 2 * scale]) & (x[scale:n - scale] > x[2 * scale:])
    populated = local.sum(axis=1)
    if not populated.any():
        return np.array([], dtype=int)
    retained = int(np.argmax(populated)) + 1
    return np.flatnonzero(np.all(local[:retained], axis=0))


def detect_onsets(ppg, fs, polarity="raw_intensity"):
    """Return onset times in seconds on the supplied continuous signal's clock.

``fs`` must exceed 16 Hz (the fixed bandpass requires this); 64 and 400 Hz are
supported directly. Nonfinite inputs are rejected so failure coverage cannot
silently disappear. Flatlines and segments shorter than two seconds yield an
empty event sequence. Disjoint clean regions must not be concatenated before
calling this function. No learned or annotation-selected parameter is used.
"""
    x = np.asarray(ppg)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("PPG must be a one-dimensional real numeric signal")
    if not np.isfinite(fs) or fs <= 16:
        raise ValueError("fs must be finite and greater than 16 Hz")
    if polarity not in ("raw_intensity", "blood_volume"):
        raise ValueError("Polarity must be specified from the acquisition convention")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite PPG: report invalid acquisition coverage")
    if len(x) < int(np.ceil(2 * fs)):
        return np.array([], dtype=float)
    x = np.asarray(x, dtype=np.float64).copy()
    scale = np.max(np.abs(x))
    if scale:
        x /= scale
    x -= x.mean()
    deviation = x.std()
    if deviation == 0:
        return np.array([], dtype=float)
    x /= deviation
    if polarity == "blood_volume":
        x = -x
    filtered = sosfiltfilt(butter(3, [.5, 8.], btype="bandpass", fs=fs, output="sos"), x)
    window = int(np.floor(6 * fs + .5))
    step = int(np.floor(window * .8 + .5))
    factor = max(1, int(np.floor(fs / 20.)))
    reduced_fs = fs / factor
    tolerance = int(np.ceil(.05 * fs))
    indices = []
    for start in _window_starts(len(filtered), window, step):
        segment = filtered[start:min(len(filtered), start + window)]
        coarse = _multiscale_maxima(segment[::factor], reduced_fs)
        for index in coarse * factor:
            low, high = max(0, int(index) - tolerance), min(len(segment), int(index) + tolerance + 1)
            index = low + int(np.argmax(segment[low:high]))
            indices.append(start + index)
    return np.unique(np.asarray(indices, dtype=np.int64)) / float(fs)


msptdfast_onsets = detect_onsets
