"""Timestamp-based resampling onto a uniform grid.

Phones and webcams deliver jittery, sometimes dropped frames; never assume constant fps.
Signals are linearly interpolated at the grid instants; a validity mask is carried along so that
grid samples bracketed by invalid or missing frames are flagged.
"""

from __future__ import annotations

import numpy as np

MAX_GAP_S = 0.25  # grid samples inside a longer gap between valid frames are invalid


def uniform_grid(t0: float, t1: float, fs: float) -> np.ndarray:
    n = int(np.floor((t1 - t0) * fs + 1e-9)) + 1
    return t0 + np.arange(n) / fs


def resample(
    t: np.ndarray,
    x: np.ndarray,
    fs: float = 30.0,
    valid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    max_gap_s: float = MAX_GAP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample x(t) along axis 0 onto a uniform grid.

    Args:
        t: [N] strictly increasing timestamps (s).
        x: [N, ...] samples.
        fs: target rate (Hz). Ignored if `t_grid` is given.
        valid: optional [N, ...] boolean mask broadcastable to x (e.g. [N, P] for x [N, P, 3]).
        max_gap_s: grid points inside a gap between consecutive valid samples longer than this
            are marked invalid.

    Returns:
        (t_grid [M], x_grid [M, ...], valid_grid [M, ...] matching `valid`'s trailing shape).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x)
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must be 1-D with len(t) == x.shape[0]")
    if np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        keep = np.concatenate([[True], np.diff(t[order]) > 0])
        t, x = t[order][keep], x[order][keep]
        if valid is not None:
            valid = np.asarray(valid)[order][keep]
    tg = uniform_grid(t[0], t[-1], fs) if t_grid is None else np.asarray(t_grid, dtype=np.float64)

    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    vmask_full = None
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        vshape = v.shape[1:]
        vmask_full = np.broadcast_to(v.reshape(v.shape + (1,) * (x.ndim - v.ndim)), x.shape).reshape(
            x.shape[0], -1
        )

    out = np.empty((tg.shape[0], flat.shape[1]), dtype=np.float64)
    out_valid = np.ones_like(out, dtype=bool)
    for j in range(flat.shape[1]):
        if vmask_full is None:
            tj, xj = t, flat[:, j]
        else:
            m = vmask_full[:, j]
            if m.sum() < 2:
                out[:, j] = np.nan
                out_valid[:, j] = False
                continue
            tj, xj = t[m], flat[m, j]
        out[:, j] = np.interp(tg, tj, xj)
        # mark grid points outside the sampled range or inside long gaps
        idx = np.searchsorted(tj, tg, side="right")  # tj[left] <= tg < tj[right]
        left = np.clip(idx - 1, 0, len(tj) - 1)
        right = np.clip(idx, 0, len(tj) - 1)
        gap = tj[right] - tj[left]
        on_sample = np.abs(tg - tj[left]) <= 1e-9
        bad = (tg < tj[0] - 1e-9) | (tg > tj[-1] + 1e-9) | ((gap > max_gap_s) & ~on_sample)
        out_valid[:, j] = ~bad

    x_grid = out.reshape((tg.shape[0],) + x.shape[1:])
    v_grid = out_valid.reshape((tg.shape[0],) + x.shape[1:])
    if valid is not None:
        # collapse back to the mask's own trailing shape (e.g. [M, P] from [M, P, 3])
        extra = x.ndim - np.asarray(valid).ndim
        if extra > 0:
            v_grid = v_grid.all(axis=tuple(range(-extra, 0)))
        v_grid = v_grid.reshape((tg.shape[0],) + vshape)
    else:
        v_grid = v_grid.all(axis=tuple(range(1, x.ndim))) if x.ndim > 1 else v_grid
    return tg, x_grid, v_grid
