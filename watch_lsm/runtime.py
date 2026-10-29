"""Explicit device selection; automatic CPU fallback is never used for cuda requests."""

import random

import numpy as np
import torch

from .paths import machine_config


def select_device(requested=None):
    name = requested or machine_config().get("runtime", {}).get("device", "auto")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Install the CUDA wheel and check the driver, or select --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable on this machine")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device: {name}")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
