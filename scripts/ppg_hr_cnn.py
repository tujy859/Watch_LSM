"""Small supervised PPG-only HR control for the phase-one evaluation PoC.

This is a new, fixed task baseline, not a pretrained foundation model. The
recipe is declared before evaluation: 8 s / 64 Hz input, four strided Conv1d
blocks, GroupNorm, a 32-unit head, Adam (1e-3), weighted L1 loss, and at most
20 epochs. The loss gives each training subject equal total weight; model
selection uses the same subject-macro MAE on validation subjects.

Only window-local z-scoring is applied. No dataset-level statistics or test
labels are used. Selection and a subsequent all-training-subject refit are
separate calls: set ``fixed_epochs=selection_metadata['best_epoch']`` and omit
all validation arguments for the refit. Input arrays are copied once to the
chosen device; a DataLoader and additional worker processes are unnecessary.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn


INPUT_SAMPLES = 512
TARGET_CENTER = 100.0
TARGET_SCALE = 50.0
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20


class SmallHRCNN(nn.Module):
    """Approximately 27k parameters; accepts already normalized [N,1,512]."""

    def __init__(self):
        super().__init__()
        layers = []
        for in_channels, out_channels, kernel in (
            (1, 16, 9), (16, 24, 7), (24, 32, 5), (32, 48, 5)
        ):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel, stride=2,
                          padding=kernel // 2),
                nn.GroupNorm(4 if out_channels == 16 else 8, out_channels),
                nn.ReLU(),
            ])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(8))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(48 * 8, 32), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scaled HR; public prediction helpers convert back to bpm."""
        return self.head(self.features(x)).squeeze(-1)


def _windows(x, device: torch.device, name: str) -> torch.Tensor:
    # torch.tensor copies NumPy memmaps, avoiding writable-view assumptions.
    if isinstance(x, torch.Tensor):
        tensor = x.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (1, INPUT_SAMPLES):
        raise ValueError(f"{name} must have shape [N,512] or [N,1,512]")
    return tensor


def normalize_windows(x: torch.Tensor) -> torch.Tensor:
    """Per-window population z-score, with flat windows retained as zeros."""
    centered = x - x.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def _subjects(subjects, n: int, name: str) -> np.ndarray:
    if isinstance(subjects, torch.Tensor):
        subjects = subjects.detach().cpu().numpy()
    raw = np.asarray(subjects)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError(f"{name} must be a length-{n} vector")
    if any(v is None or str(v).strip() in ("", "nan", "None") for v in raw):
        raise ValueError(f"{name} contains a missing subject identity")
    return raw.astype(str)


def subject_weights(subjects) -> np.ndarray:
    """Weights have mean 1 and each subject has identical summed weight."""
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or not len(subjects):
        raise ValueError("A nonempty vector of subject identities is required")
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return (len(subjects) / (len(counts) * counts[inverse])).astype(np.float32)


def _prepare_labeled(x, y, subjects, device, name):
    x = _windows(x, device, name + "_x")
    if not len(x):
        raise ValueError(f"{name} contains no windows")
    y = torch.as_tensor(y, dtype=torch.float32, device=device).detach()
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError(f"{name}_y must be a length-{len(x)} vector")
    if not torch.isfinite(x).all().item():
        raise ValueError(f"{name}_x contains nonfinite samples; select valid rows explicitly")
    if not torch.isfinite(y).all().item() or not (y > 0).all().item():
        raise ValueError(f"{name}_y must contain finite positive reference HR")
    subjects = _subjects(subjects, len(x), name + "_subjects")
    weights = torch.as_tensor(subject_weights(subjects), device=device)
    return normalize_windows(x), (y - TARGET_CENTER) / TARGET_SCALE, subjects, weights


@torch.inference_mode()
def _validation_mae(model, x, y, weights, batch_size):
    error = torch.zeros((), device=x.device)
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        pred = model(x[start:stop])
        error += ((pred - y[start:stop]).abs() * weights[start:stop]).sum()
    value = float((error * TARGET_SCALE / len(x)).item())
    if not np.isfinite(value):
        raise FloatingPointError("CNN validation produced a nonfinite MAE")
    return value


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Independent CPU tensors suitable for torch.save(weights_only loading)."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_cnn(
    train_x, train_y, train_subjects, *,
    val_x=None, val_y=None, val_subjects=None,
    seed: int = 42, device: str | torch.device = "cpu",
    max_epochs: int = MAX_EPOCHS, batch_size: int = 1024, patience: int = 5,
    fixed_epochs: int | None = None,
) -> tuple[SmallHRCNN, dict[str, Any]]:
    """Fit using validation selection, or refit for a preselected epoch count.

    All three validation arguments are required for selection. Validation
    subjects must be disjoint from training subjects. In ``fixed_epochs`` mode
    all validation arguments must be omitted, and the final epoch is returned.
    The caller owns the outer test split; no test information enters fitting.
    Nonfinite inputs fail explicitly instead of silently changing coverage.
    """
    if not (1 <= max_epochs <= MAX_EPOCHS) or batch_size < 1 or patience < 1:
        raise ValueError("Require 1 <= max_epochs <= 20 and positive batch_size/patience")
    has_val = [value is not None for value in (val_x, val_y, val_subjects)]
    if fixed_epochs is None and not all(has_val):
        raise ValueError("Validation selection requires x, y, and subject identities")
    if fixed_epochs is not None:
        if any(has_val):
            raise ValueError("Fixed-epoch refitting must not receive validation data")
        if not (1 <= fixed_epochs <= max_epochs):
            raise ValueError("fixed_epochs must be in [1,max_epochs]")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_x, train_y, train_ids, weights = _prepare_labeled(
        train_x, train_y, train_subjects, device, "train"
    )
    validation = None
    val_ids = np.array([], dtype=str)
    if fixed_epochs is None:
        validation = _prepare_labeled(val_x, val_y, val_subjects, device, "validation")
        val_ids = validation[2]
        if set(train_ids) & set(val_ids):
            raise ValueError("Training and validation subjects overlap")
    epoch_limit = fixed_epochs if fixed_epochs is not None else max_epochs
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    start_time = time.monotonic()
    history = []
    best_epoch, best_mae, best_state = 0, float("inf"), None
    stale_epochs = 0
    # Restore the caller's random state and cuDNN options after each fit. FP32
    # avoids the known local cuDNN/FP16 failure in the unrelated FM pipeline.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            model = SmallHRCNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            for epoch in range(1, epoch_limit + 1):
                model.train()
                permutation = torch.randperm(len(train_x), device=device)
                total_loss = torch.zeros((), device=device)
                for start in range(0, len(train_x), batch_size):
                    indices = permutation[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(train_x[indices])
                    loss = ((pred - train_y[indices]).abs() * weights[indices]).mean()
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("CNN training produced a nonfinite loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                             error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += loss.detach() * len(indices)
                model.eval()
                record = {"epoch": epoch, "training_weighted_mae_bpm":
                          float((total_loss * TARGET_SCALE / len(train_x)).item())}
                if validation is not None:
                    vx, vy, _, vw = validation
                    val_mae = _validation_mae(model, vx, vy, vw, batch_size)
                    record["validation_subject_mae_bpm"] = val_mae
                    if val_mae < best_mae:
                        best_epoch, best_mae = epoch, val_mae
                        best_state = cpu_state_dict(model)
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                else:
                    best_epoch = epoch
                history.append(record)
                if validation is not None and stale_epochs >= patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state, strict=True)
            model.eval()
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
    metadata = {
        "model": "small-cnn-v1", "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": "Conv1d channels 16/24/32/48, kernels 9/7/5/5, stride 2; GroupNorm/ReLU; adaptive average pool 8; Linear 384->32->1; dropout 0.1",
        "input_sampling_rate_hz": 64, "window_seconds": 8,
        "normalization": "per-window population z-score; std floor 1e-6",
        "target_center_bpm": TARGET_CENTER, "target_scale_bpm": TARGET_SCALE,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "loss": "subject-balanced weighted L1", "gradient_clip_norm": 5.0,
        "precision": "float32", "seed": int(seed), "device": str(device),
        "batch_size": int(batch_size), "max_epochs": int(max_epochs),
        "patience": int(patience), "fixed_epochs": fixed_epochs,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "best_validation_subject_mae_bpm": best_mae if validation is not None else None,
        "train_subjects": sorted(set(train_ids.tolist())),
        "validation_subjects": sorted(set(val_ids.tolist())),
        "train_windows": len(train_x),
        "validation_windows": len(validation[0]) if validation is not None else 0,
        "history": history, "elapsed_seconds": time.monotonic() - start_time,
    }
    return model, metadata


@torch.inference_mode()
def predict_cnn(model: SmallHRCNN, x, batch_size: int = 1024) -> np.ndarray:
    """Return bpm with preserved row order; nonfinite input rows return NaN."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    x = _windows(x, device, "prediction_x")
    predictions = np.full(len(x), np.nan, dtype=np.float32)
    model.eval()
    for start in range(0, len(x), batch_size):
        block = x[start:start + batch_size]
        valid = torch.isfinite(block).all(dim=(1, 2))
        if valid.any().item():
            pred = model(normalize_windows(block[valid])) * TARGET_SCALE + TARGET_CENTER
            if not torch.isfinite(pred).all().item():
                raise FloatingPointError("CNN predictions are nonfinite for finite inputs")
            positions = np.flatnonzero(valid.cpu().numpy()) + start
            predictions[positions] = pred.cpu().numpy()
    return predictions


def fit_predict(train_x, train_y, train_subjects, val_x, val_y, val_subjects,
                test_x, *, seed=42, device="cpu", max_epochs=20, batch_size=1024):
    """Convenience wrapper; returns predictions, fit metadata, and CPU weights."""
    model, metadata = fit_cnn(
        train_x, train_y, train_subjects, val_x=val_x, val_y=val_y,
        val_subjects=val_subjects, seed=seed, device=device,
        max_epochs=max_epochs, batch_size=batch_size,
    )
    return predict_cnn(model, test_x), metadata, cpu_state_dict(model)
