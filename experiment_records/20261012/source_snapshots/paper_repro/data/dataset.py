"""
Dataset (paper AAS74510R3). Mirrors train_paper_fidelity.py FRBDataset:
  (15360,1,1024) -> (1,15360,1024), clip(-5,5), global z-score,
  SpecAugment-style time/frequency masking + noise + brightness (train only).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class FRBDataset(Dataset):
    def __init__(self, pos_files, neg_files, cfg, augment=False):
        self.files = pos_files + neg_files
        self.labels = [1] * len(pos_files) + [0] * len(neg_files)
        self.cfg = cfg
        self.augment = augment
        self.valid = [(f, l) for f, l in zip(self.files, self.labels) if os.path.exists(f)]
        if len(self.valid) < len(self.files):
            print(f"  Warning: {len(self.files) - len(self.valid)} files missing")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        f, label = self.valid[idx]
        try:
            data = np.load(f, mmap_mode="r")
            data = np.array(data, dtype=np.float32)
            if data.ndim == 3:
                if data.shape[0] == 1:
                    data = data
                elif data.shape[1] == 1:
                    data = data[:, 0, :]
                    data = data[np.newaxis, :, :]
                else:
                    data = data.mean(axis=0, keepdims=True)
            elif data.ndim == 2:
                data = data[np.newaxis, :, :]

            data = np.clip(data, -5, 5)
            if self.cfg.get("norm_mode", "global") == "channel":
                # per-frequency-channel z-score along time axis (v15 breakthrough:
                # removes stable narrowband RFI baseline per channel)
                mean = data.mean(axis=1, keepdims=True)   # (1,1,F)
                std = data.std(axis=1, keepdims=True)
                data = (data - mean) / (std + 1e-8)
            else:
                mean = data.mean()
                std = data.std()
                if std > 1e-8:
                    data = (data - mean) / (std + 1e-8)

            if self.augment and np.random.rand() < self.cfg.get("augment_prob", 0.8):
                data = self._augment(data)

            data = np.ascontiguousarray(data)
            return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)
        except Exception:
            return torch.zeros(1, self.cfg["n_time"], self.cfg["n_freq"]), \
                torch.tensor(0, dtype=torch.long)

    def _augment(self, d):
        d = d.copy()
        _, T, F = d.shape
        if np.random.rand() > 0.5:
            fm = np.random.randint(0, self.cfg.get("freq_mask_param", 30))
            f0 = np.random.randint(0, max(1, F - fm))
            d[:, :, f0:f0 + fm] = 0
        if np.random.rand() > 0.5:
            tm = np.random.randint(0, self.cfg.get("time_mask_param", 50))
            t0 = np.random.randint(0, max(1, T - tm))
            d[:, t0:t0 + tm, :] = 0
        if np.random.rand() > 0.7:
            noise = np.random.randn(*d.shape).astype(np.float32) * 0.03
            d = d + noise
        if np.random.rand() > 0.7:
            factor = np.random.uniform(0.9, 1.1)
            d = d * factor
        return d
