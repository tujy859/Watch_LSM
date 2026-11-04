"""Small dense task heads for ECG-referenced cardiac events inferred from PPG.

The AnyPPG encoder is frozen, including batch-normalization statistics. Its six
temporal stages are retained instead of the pooled embedding. Neither model's
output is a manually annotated PPG systolic peak. Both predict a soft ECG R-event
target; timing uncertainty and PRV/RR differences are evaluated separately.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import butter, sosfiltfilt

STAGE_CHANNELS = (64, 160, 160, 400, 400, 512)
OUTPUT_FS = 64
INPUT_SECONDS = 8
OUTPUT_SAMPLES = 512
TARGET_SIGMA_SECONDS = 0.04
THRESHOLDS = (0.2, 0.35, 0.5, 0.65)


def normalize(x):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def anyppg_input(x):
    """Official 125 Hz, 0.5–8 Hz filtering and window-wise normalization."""
    x = np.asarray(x, dtype=np.float32)
    sos = butter(3, [0.5, 8.], btype="bandpass", fs=125, output="sos")
    x = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    mean, std = x.mean(axis=-1, keepdims=True), x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)


@torch.no_grad()
def frozen_stages(encoder, x):
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Encoder must be frozen and in evaluation mode")
    out = encoder.first_activation(encoder.first_bn(encoder.first_conv(x)))
    outputs = []
    for stage in encoder.stage_list:
        out = stage(out)
        outputs.append(out)
    return outputs


class AnyPPGEventHead(nn.Module):
    """Six learned pointwise projections and a small temporal decoder."""
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, 8, 1), nn.GroupNorm(4, 8), nn.ReLU())
            for channels in STAGE_CHANNELS
        ])
        self.decoder = nn.Sequential(
            nn.Conv1d(48, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, stages):
        projected = [F.interpolate(layer(x), size=OUTPUT_SAMPLES, mode="linear",
                                   align_corners=False)
                     for layer, x in zip(self.projections, stages)]
        return self.decoder(torch.cat(projected, dim=1)).squeeze(1)


class SmallEventCNN(nn.Module):
    """A fixed, randomly initialized dense CNN with about 20k parameters."""
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout, kernel, dilation in (
            (1, 16, 9, 1), (16, 24, 7, 1), (24, 32, 5, 2),
            (32, 32, 5, 4), (32, 32, 5, 8),
        ):
            layers += [nn.Conv1d(cin, cout, kernel, padding=(kernel // 2) * dilation,
                                 dilation=dilation), nn.GroupNorm(4, cout), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Conv1d(32, 1, 1))

    def forward(self, x):
        return self.net(normalize(x)).squeeze(1)


def soft_events(r_times, starts):
    """Gaussian event targets at 64 Hz, preserving the original ECG time axis."""
    starts = np.asarray(starts, dtype=float)
    r_times = np.asarray(r_times, dtype=float)
    y = np.zeros((len(starts), OUTPUT_SAMPLES), np.float32)
    relative = np.arange(OUTPUT_SAMPLES) / OUTPUT_FS
    for i, start in enumerate(starts):
        relevant = r_times[(r_times >= start - 0.2) & (r_times < start + 8.2)]
        for event in relevant:
            y[i] = np.maximum(y[i], np.exp(-0.5 * ((relative - (event - start)) /
                                                  TARGET_SIGMA_SECONDS) ** 2))
    return y


def parameter_counts():
    return {name: sum(p.numel() for p in model.parameters()) for name, model in (
        ("AnyPPG_head", AnyPPGEventHead()), ("SmallCNN", SmallEventCNN()))}
