"""Live-inference input assembly: ring buffer of raw frames -> one model window.

Single source of truth for the desktop demo, the golden files and (by port) the Android app:
    1. resample the whole buffer onto a uniform grid starting at the buffer's first timestamp
       (`pulsar.preprocess.resample.resample`, linear interpolation, gap-aware validity);
    2. keep the last T = win_s * fs grid samples; invalid/NaN samples -> 0 (the model masks them);
    3. n_pix and motion = means over the last T *raw* frames (NaN-aware).
"""

from __future__ import annotations

import numpy as np

from pulsar.preprocess.resample import MAX_GAP_S, resample

BUFFER_EXTRA_S = 2.0  # ring buffer keeps win_s + this, so the grid's first samples are not edge-clamped


def window_contract(fs: float, win_s: float) -> dict:
    """Live-window parameters a port must mirror (model_meta.json "window")."""
    return {
        "fs_hz": fs,
        "win_s": win_s,
        "t_len": int(round(win_s * fs)),
        "max_gap_s": MAX_GAP_S,
        "buffer_s": win_s + BUFFER_EXTRA_S,
        "grid": "uniform from the buffer's first timestamp; keep the last t_len samples",
        "invalid_fill": 0.0,
        "n_pix_motion": "NaN-aware mean over the last t_len raw frames",
    }


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
