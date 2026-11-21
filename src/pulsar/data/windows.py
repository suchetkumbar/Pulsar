"""Fixed-length analysis windows on a uniform 30 Hz grid (docs 04 §1, 06 §1).

A window holds per-patch linear-RGB traces, validity, the aligned GT BVP, and the window's HR
target (spectral peak of the GT on the exact bpm grid -- same estimator as predictions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar.data.bundle import TraceBundle
from pulsar.data.gt import gt_on_grid
from pulsar.preprocess.resample import resample
from pulsar.signal.filters import HR_BAND_HZ, bandpass
from pulsar.signal.spectral import bpm_grid, hr_grid, snr_db

SPO2_VALID = (50.0, 100.0)  # CMS50E-style dropout codes (0, 127, ...) fall outside this range


def window_spo2(b: TraceBundle, t0: float, t1: float) -> float:
    """Mean reference SpO2 over face-time [t0, t1] (GT clock shifted by the lag); NaN on dropout."""
    if b.gt_spo2 is None:
        return float("nan")
    m = (b.gt_t >= t0 + b.lag_s) & (b.gt_t <= t1 + b.lag_s)
    v = np.asarray(b.gt_spo2, dtype=np.float64)[m]
    if v.size == 0 or not np.all(np.isfinite(v)) or v.min() < SPO2_VALID[0] or v.max() > SPO2_VALID[1]:
        return float("nan")
    return float(v.mean())


@dataclass
class Windows:
    x: np.ndarray  # [N, T, P, 3] float32 linear RGB
    valid: np.ndarray  # [N, T, P] bool
    gt_bvp: np.ndarray  # [N, T] float32 (band-passed, z-scored)
    hr: np.ndarray  # [N] bpm target
    gt_snr: np.ndarray  # [N] dB, quality of the GT itself
    start_s: np.ndarray  # [N]
    n_pix: np.ndarray  # [N, P] mean pixel count
    motion: np.ndarray  # [N] mean landmark motion
    center: np.ndarray  # [N, P, 2] mean normalised patch centre (x, y)
    video_id: str
    subject: str
    meta: dict
    spo2: np.ndarray | None = None  # [N] window-mean SpO2 (%), NaN on oximeter dropout; None if no GT

    def __len__(self) -> int:
        return int(self.hr.shape[0])

    def face_mean_rgb(self, min_valid_frac: float = 0.8) -> np.ndarray:
        """Pixel-count-weighted mean over valid patches -> [N, T, 3] (for classical baselines)."""
        w = self.valid.astype(np.float64) * self.n_pix[:, None, :]
        keep = self.valid.mean(axis=1) >= min_valid_frac  # [N, P]
        w = w * keep[:, None, :]
        num = np.einsum("ntp,ntpc->ntc", w, np.nan_to_num(self.x.astype(np.float64)))
        den = w.sum(axis=2, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den


def make_windows(
    b: TraceBundle,
    fs: float = 30.0,
    win_s: float = 10.0,
    stride_s: float = 1.0,
    band: tuple[float, float] = HR_BAND_HZ,
    grid_bpm: np.ndarray | None = None,
    min_valid_frac: float = 0.5,
) -> Windows:
    grid_bpm = bpm_grid() if grid_bpm is None else grid_bpm
    face = b.face_found
    tg, x, v = resample(b.t, b.rgb_lin, fs=fs, valid=b.valid & face[:, None])
    _, npx, _ = resample(b.t, b.n_pix.astype(np.float64), fs=fs, t_grid=tg)
    _, mot, _ = resample(b.t, b.motion.astype(np.float64), fs=fs, t_grid=tg)
    _, cen, _ = resample(b.t, np.nan_to_num(b.center, nan=0.5).astype(np.float64), fs=fs, t_grid=tg)
    # restrict to the interval covered by ground truth (after lag)
    gt_lo, gt_hi = b.gt_t[0] - b.lag_s, b.gt_t[-1] - b.lag_s
    gt_full = gt_on_grid(b, tg)

    t_len = int(round(win_s * fs))
    step = max(1, int(round(stride_s * fs)))
    xs, vs, gts, hrs, snrs, starts, nps, mots, cens, sats = [], [], [], [], [], [], [], [], [], []
    for s in range(0, len(tg) - t_len + 1, step):
        sl = slice(s, s + t_len)
        if tg[s] < gt_lo or tg[s + t_len - 1] > gt_hi:
            continue
        vw = v[sl]
        if vw.mean() < min_valid_frac:
            continue
        g = bandpass(gt_full[sl], fs, band)
        if g.std() < 1e-9:
            continue
        g = (g - g.mean()) / g.std()
        hr = hr_grid(g, fs, grid_bpm)
        xs.append(np.nan_to_num(x[sl]).astype(np.float32))
        vs.append(vw)
        gts.append(g.astype(np.float32))
        hrs.append(hr)
        snrs.append(snr_db(g, hr, fs, band))
        starts.append(tg[s] - tg[0])
        nps.append(npx[sl].mean(axis=0))
        mots.append(mot[sl].mean())
        cens.append(cen[sl].mean(axis=0))
        sats.append(window_spo2(b, tg[s], tg[s + t_len - 1]))
    p_n = b.num_patches
    if not hrs:
        empty = np.zeros((0,))
        return Windows(
            np.zeros((0, t_len, p_n, 3), np.float32),
            np.zeros((0, t_len, p_n), bool),
            np.zeros((0, t_len), np.float32),
            empty,
            empty,
            empty,
            np.zeros((0, p_n)),
            empty,
            np.zeros((0, p_n, 2)),
            b.meta.get("video_id", ""),
            b.meta.get("subject", ""),
            dict(b.meta),
            None if b.gt_spo2 is None else empty,
        )
    return Windows(
        x=np.stack(xs),
        valid=np.stack(vs),
        gt_bvp=np.stack(gts),
        hr=np.asarray(hrs),
        gt_snr=np.asarray(snrs),
        start_s=np.asarray(starts),
        n_pix=np.stack(nps),
        motion=np.asarray(mots),
        center=np.stack(cens).astype(np.float32),
        video_id=str(b.meta.get("video_id", "")),
        subject=str(b.meta.get("subject", "")),
        meta=dict(b.meta),
        spo2=None if b.gt_spo2 is None else np.asarray(sats),
    )
