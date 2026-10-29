import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.signal
import pandas as pd
from .normalizer import robust_zscore
from .augmentations import PhysiologicalAugmenter

class SmartwatchDataset(Dataset):
    """
    Modular Multi-Rate Dataset for Smartwatch Physiological Signal Pretraining and Evaluation.
    """
    def __init__(
        self,
        ppg_list,
        acc_list,
        hr_list=None,
        cadence_list=None,
        activity_list=None,
        orig_fs=256,
        target_fs_ppg=64,
        target_fs_acc=32,
        window_sec=8.0,
        step_sec=2.0,
        is_training=True,
        augment=False
    ):
        self.target_fs_ppg = target_fs_ppg
        self.target_fs_acc = target_fs_acc
        self.ppg_len = int(window_sec * target_fs_ppg) # e.g. 512
        self.acc_len = int(window_sec * target_fs_acc)  # e.g. 256
        self.is_training = is_training
        self.augment = augment
        self.augmenter = PhysiologicalAugmenter() if augment else None
        
        self.samples = []
        
        # Iterate over recordings
        for i in range(len(ppg_list)):
            p_sig = ppg_list[i]
            a_sig = acc_list[i]
            hr_sig = hr_list[i] if hr_list is not None else None
            cad_sig = cadence_list[i] if cadence_list is not None else None
            act_sig = activity_list[i] if activity_list is not None else None
            
            # Slice sliding windows
            win_pts = int(window_sec * orig_fs)
            step_pts = int(step_sec * orig_fs)
            
            total_pts = len(p_sig)
            for start in range(0, total_pts - win_pts + 1, step_pts):
                end = start + win_pts
                p_win = p_sig[start:end]
                a_win = a_sig[start:end]
                
                # Resample PPG to target_fs_ppg
                p_resamp = scipy.signal.resample(p_win, self.ppg_len)
                # Resample ACC to target_fs_acc
                a_resamp = scipy.signal.resample(a_win, self.acc_len, axis=0)
                
                # Normalize
                p_norm = robust_zscore(p_resamp)
                a_norm = robust_zscore(a_resamp, axis=0)
                
                # Optional target labels
                sample_dict = {
                    "ppg": p_norm.reshape(1, self.ppg_len).astype(np.float32),
                    "acc": a_norm.T.astype(np.float32) # (3, acc_len)
                }
                
                if hr_sig is not None:
                    sample_dict["hr"] = np.float32(hr_sig[start // orig_fs]) if start // orig_fs < len(hr_sig) else np.float32(75.0)
                if cad_sig is not None:
                    sample_dict["cadence"] = np.float32(cad_sig[start // orig_fs]) if start // orig_fs < len(cad_sig) else np.float32(0.0)
                if act_sig is not None:
                    sample_dict["activity"] = int(act_sig)
                    
                self.samples.append(sample_dict)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        ppg = item["ppg"].copy()
        acc = item["acc"].copy()
        
        if self.augment and self.augmenter is not None:
            ppg, acc = self.augmenter(ppg, acc, fs=self.target_fs_ppg)
            
        data_out = {
            "ppg": torch.tensor(ppg, dtype=torch.float32),
            "acc": torch.tensor(acc, dtype=torch.float32)
        }
        if "hr" in item:
            data_out["hr"] = torch.tensor(item["hr"], dtype=torch.float32)
        if "cadence" in item:
            data_out["cadence"] = torch.tensor(item["cadence"], dtype=torch.float32)
        if "activity" in item:
            data_out["activity"] = torch.tensor(item["activity"], dtype=torch.long)
            
        return data_out
