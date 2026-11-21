"""Training-time augmentation pipeline: standard + PhysTone-Aug (C1) + IllumSim-IN (C2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pulsar.aug import illumsim, phystone


@dataclass
class AugConfig:
    standard: bool = True
    gain_range: tuple[float, float] = (0.7, 1.3)
    patch_dropout: float = 0.15  # max fraction of patches randomly invalidated
    naive_tone_p: float = 0.0  # ablation A2
    phystone: phystone.PhysToneConfig | None = None  # C1
    illumsim: illumsim.IllumSimConfig | None = None  # C2
    illumsim_p: float = 0.5
    extra: dict = field(default_factory=dict)


def apply(
    x: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    n_pix: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    cfg: AugConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """x [T, P, 3] linear RGB, valid [T, P] -> augmented (x, valid, info)."""
    info: dict = {}
    x = x.astype(np.float64, copy=True)
    valid = valid.copy()
    if cfg.naive_tone_p > 0 and rng.random() < cfg.naive_tone_p:
        x = phystone.naive_tone(x, rng).astype(np.float64)
        info["naive_tone"] = True
    if cfg.phystone is not None and rng.random() < cfg.phystone.p_apply:
        x, pinfo = phystone.augment(x, n_pix, rng, cfg.phystone, valid=valid)
        x = x.astype(np.float64)
        info["phystone"] = pinfo
    if cfg.illumsim is not None and rng.random() < cfg.illumsim_p:
        x, sinfo = illumsim.simulate(x, center, n_pix, fs, rng, cfg.illumsim)
        x = x.astype(np.float64)
        info["illumsim"] = sinfo.components
    if cfg.standard:
        x = x * rng.uniform(*cfg.gain_range)
        if cfg.patch_dropout > 0:
            p_n = x.shape[1]
            k = rng.integers(0, int(cfg.patch_dropout * p_n) + 1)
            if k:
                valid[:, rng.choice(p_n, k, replace=False)] = False
    return np.clip(x, 0, None).astype(np.float32), valid, info
