"""Live-inference input assembly: ring buffer of raw frames -> one model window.

Single source of truth for the desktop demo, the golden files and (by port) the Android app:
    1. resample the whole buffer onto a uniform grid starting at the buffer's first timestamp
       (`pulsar.preprocess.resample.resample`, linear interpolation, gap-aware validity);
    2. keep the last T = win_s * fs grid samples; invalid/NaN samples -> 0 (the model masks them);
    3. n_pix and motion = means over the last T *raw* frames (NaN-aware).
"""

from __future__ import annotations

import numpy as np

from pulsar.preprocess.resample import resample


def live_window(
    t: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    n_pix: np.ndarray,
    motion: np.ndarray,
    fs: float = 30.0,
    win_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Raw buffer t [N], rgb [N, P, 3], valid [N, P], n_pix [N, P], motion [N] ->
    (t_grid [T], x [T, P, 3] float32, valid [T, P] bool, n_pix [P] float32, motion), or None if too short."""
    tg, x, v = resample(np.asarray(t, float), np.asarray(rgb), fs=fs, valid=np.asarray(valid, bool))
    n = int(round(win_s * fs))
    if len(tg) < n:
        return None
    x = np.nan_to_num(x[-n:]).astype(np.float32)
    v = v[-n:]
    with np.errstate(invalid="ignore"):
        npx = np.nan_to_num(np.nanmean(np.asarray(n_pix, float)[-n:], axis=0)).astype(np.float32)
        mot = float(np.nan_to_num(np.nanmean(np.asarray(motion, float)[-n:])))
    return tg[-n:], x, v, npx, mot
