"""
Watch-LSM: Lightweight Smartwatch Physiological Signal Foundation Model Library
Designed for PPG, 3-Axis Accelerometer, Heart Rate, and Step Count Analytics.
"""

__version__ = "0.2.0"
__all__ = [
    "MultiRatePatchTokenizer",
    "WatchLSMBackbone",
    "WatchLSMModel",
    "HeartRateHead",
    "StepCountHead",
    "ActivityClassificationHead",
    "SmartwatchDataset"
]


def __getattr__(name):
    """Keep path/config utilities usable without importing the training stack."""
    from importlib import import_module
    modules = {
        "MultiRatePatchTokenizer": ".models.tokenizer",
        "WatchLSMBackbone": ".models.backbone",
        "WatchLSMModel": ".models.backbone",
        "HeartRateHead": ".models.heads",
        "StepCountHead": ".models.heads",
        "ActivityClassificationHead": ".models.heads",
        "SmartwatchDataset": ".data.dataset",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
