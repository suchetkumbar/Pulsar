"""Filtering and detrending for rPPG signals.

`detrend_tarvainen` and `bandpass_toolbox` reproduce rPPG-Toolbox's post-processing exactly
(smoothness-priors detrend with lambda=100, 1st-order Butterworth 0.75-2.5 Hz, filtfilt), so our
baseline numbers are comparable. `bandpass` is our default (SOS, configurable order/band).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import sparse
from scipy.sparse.linalg import spsolve

HR_BAND_HZ: tuple[float, float] = (0.7, 3.5)  # 42-210 bpm (Pulsar default)
TOOLBOX_BAND_HZ: tuple[float, float] = (0.75, 2.5)  # 45-150 bpm (rPPG-Toolbox default)


def detrend_tarvainen(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al., 2002), as used by rPPG-Toolbox.

    Solves x - (I + lam^2 D^T D)^{-1} x with a sparse second-difference operator D.
    Equivalent to the toolbox's dense implementation but O(N) memory.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return x - x.mean(axis=0)
    ones = np.ones(n - 2)
    d = sparse.diags([ones, -2 * ones, ones], [0, 1, 2], shape=(n - 2, n), format="csc")
    a = sparse.identity(n, format="csc") + (lam**2) * (d.T @ d)
    trend = spsolve(a.tocsc(), x)
    return x - trend.reshape(x.shape)


def bandpass(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = HR_BAND_HZ,
    order: int = 2,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass (second-order sections)."""
    lo, hi = band
    nyq = fs / 2.0
    if not 0 < lo < hi < nyq:
        raise ValueError(f"invalid band {band} for fs={fs}")
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    x = np.asarray(x, dtype=np.float64)
    padlen = min(3 * (2 * len(sos) + 1), x.shape[axis] - 1)
    return sps.sosfiltfilt(sos, x, axis=axis, padlen=max(padlen, 0))


def bandpass_toolbox(x: np.ndarray, fs: float) -> np.ndarray:
    """rPPG-Toolbox post-filter: butter(1, [0.75, 2.5] Hz) + filtfilt."""
    b, a = sps.butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype="bandpass")
    return sps.filtfilt(b, a, np.asarray(x, dtype=np.float64))


def postprocess_bvp(
    x: np.ndarray,
    fs: float,
    mode: str = "pulsar",
    band: tuple[float, float] = HR_BAND_HZ,
) -> np.ndarray:
    """Detrend + band-pass a raw BVP estimate.

    mode="toolbox" reproduces rPPG-Toolbox (`calculate_metric_per_video`, diff_flag=False).
    mode="pulsar" uses the same detrend with our SOS band-pass over `band`.
    """
    y = detrend_tarvainen(x, 100.0)
    if mode == "toolbox":
        return bandpass_toolbox(y, fs)
    if mode == "pulsar":
        return bandpass(y, fs, band=band, order=2)
    raise ValueError(f"unknown mode {mode!r}")
