"""
Watch-LSM: Lightweight Smartwatch Physiological Signal Foundation Model Library
Designed for PPG, 3-Axis Accelerometer, Heart Rate, and Step Count Analytics.
"""

from .models.tokenizer import MultiRatePatchTokenizer
from .models.backbone import WatchLSMBackbone, WatchLSMModel
from .models.heads import HeartRateHead, StepCountHead, ActivityClassificationHead
from .data.dataset import SmartwatchDataset

__version__ = "0.1.0"
__all__ = [
    "MultiRatePatchTokenizer",
    "WatchLSMBackbone",
    "WatchLSMModel",
    "HeartRateHead",
    "StepCountHead",
    "ActivityClassificationHead",
    "SmartwatchDataset"
]
