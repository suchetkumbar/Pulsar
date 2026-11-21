"""Ground-truth alignment: polarity and lag between the facial rPPG signal and the finger PPG.

Finger PPG lags the facial pulse (pulse transit) and the devices' clocks differ. Datasets also
disagree on waveform polarity, which would give a model contradictory BVP targets. We orient
every GT waveform so it correlates *positively* with POS (whose sign is physically fixed by its
colour projection) and estimate the lag that maximises that correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar.data.bundle import TraceBundle
from pulsar.preprocess.resample import resample
from pulsar.signal.classical import pos
from pulsar.signal.filters import bandpass


@dataclass
class Alignment:
    lag_s: float  # GT(t + lag) aligns with face(t)
    sign: int  # multiply GT by this
    corr: float  # peak correlation after alignment


def face_reference_signal(b: TraceBundle, fs: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """POS on the mean of valid patches (per-frame nanmean), on a uniform grid."""
    rgb = np.where(b.valid[..., None], b.rgb_lin, np.nan)
    with np.errstate(invalid="ignore"):
        mean_rgb = np.nanmean(rgb, axis=1)  # [T, 3]
    ok = np.isfinite(mean_rgb).all(axis=1)
    tg, x, _ = resample(b.t[ok], mean_rgb[ok], fs=fs)
    return tg, pos(x, fs)


def estimate_alignment(
    b: TraceBundle, fs: float = 30.0, max_lag_s: float = 0.5, band=(0.7, 3.5)
) -> Alignment:
    tg, ref = face_reference_signal(b, fs)
    ref = bandpass(ref, fs, band)
    ref = (ref - ref.mean()) / (ref.std() + 1e-12)
    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, 1.0 / fs)
    best = Alignment(0.0, 1, -np.inf)
    for lag in lags:
        g = np.interp(tg + lag, b.gt_t, b.gt_ppg)
        g = bandpass(g, fs, band)
        g = (g - g.mean()) / (g.std() + 1e-12)
        c = float(np.mean(ref * g))
        for sign in (1, -1):
            if sign * c > best.corr:
                best = Alignment(float(lag), sign, sign * c)
    return best


def apply_alignment(b: TraceBundle, al: Alignment) -> TraceBundle:
    b.lag_s = al.lag_s
    b.meta["gt_sign"] = al.sign
    b.meta["gt_align_corr"] = al.corr
    return b


def gt_on_grid(b: TraceBundle, t_grid: np.ndarray) -> np.ndarray:
    """Aligned, sign-corrected GT PPG evaluated at face-clock times `t_grid`."""
    sign = int(b.meta.get("gt_sign", 1))
    return sign * np.interp(t_grid + b.lag_s, b.gt_t, b.gt_ppg)
