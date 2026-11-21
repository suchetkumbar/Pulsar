"""Torch dataset over precomputed windows, with on-the-fly augmentation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pulsar.aug.pipeline import AugConfig, apply
from pulsar.data.windows import Windows


class WindowDataset(Dataset):
    def __init__(self, windows: list[Windows], aug: AugConfig | None = None, fs: float = 30.0, seed: int = 0):
        self.windows = [w for w in windows if len(w)]
        self.index = [(i, j) for i, w in enumerate(self.windows) for j in range(len(w))]
        self.aug = aug
        self.fs = fs
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, k: int) -> dict:
        i, j = self.index[k]
        w = self.windows[i]
        x, valid = w.x[j], w.valid[j]
        if self.aug is not None:
            rng = np.random.default_rng((self.seed, self.epoch, k))
            x, valid, _ = apply(x, valid, w.center[j], w.n_pix[j], self.fs, rng, self.aug)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x.transpose(1, 0, 2))),  # [P, T, 3]
            "valid": torch.from_numpy(np.ascontiguousarray(valid.T)),  # [P, T]
            "n_pix": torch.from_numpy(w.n_pix[j].astype(np.float32)),
            "motion": torch.tensor(float(w.motion[j]), dtype=torch.float32),
            "bvp": torch.from_numpy(w.gt_bvp[j]),
            "hr": torch.tensor(float(w.hr[j]), dtype=torch.float32),
            "key": k,
        }

    def window_meta(self, k: int) -> dict:
        i, j = self.index[k]
        w = self.windows[i]
        ita = w.meta.get("ita")
        return {
            "dataset": w.meta.get("dataset"),
            "video_id": w.video_id,
            "subject": f"{w.meta.get('dataset')}:{w.subject}",
            "start_s": float(w.start_s[j]),
            "fitzpatrick": w.meta.get("fitzpatrick"),
            "light": w.meta.get("light"),
            "ita": ita,
        }
