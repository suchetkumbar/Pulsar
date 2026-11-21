"""Synthetic TraceBundles with a known pulse, for tests and model smoke-training.

Signal model per patch k, channel c (linear RGB):
    c_k(t) = DC_k,c * L(t) * (1 + A_k * w_c * p(t)) + noise
* p(t): pulse waveform with harmonics and slowly varying HR (known ground truth).
* w:    blood-volume pulse colour signature (de Haan & van Leest 2014 PBV ~ [0.33, 0.77, 0.53]).
* L(t): common-mode illumination drift (should be removed by POS/CHROM).
* noise: Poisson-Gaussian camera noise averaged over n_pix pixels: Var = (a*I + b) / n_pix.
* SpO2 (optional): lower saturation raises the red pulsatile weight, w_R *= 1 + kappa*(ref - SpO2),
  so the ratio-of-ratios R/B is linear in SpO2 (a toy model for testing the estimator only).
"""

from __future__ import annotations

import numpy as np

from pulsar.data.bundle import TraceBundle
from pulsar.preprocess.color import ita_from_linear_rgb
from pulsar.preprocess.patches import PatchSpec

PBV = np.array([0.33, 0.77, 0.53])
SKIN_PRESETS = {  # approximate linear-RGB skin reflectance under neutral light
    "light": np.array([0.50, 0.33, 0.25]),
    "medium": np.array([0.32, 0.19, 0.12]),
    "dark": np.array([0.12, 0.065, 0.04]),
}


def pulse_waveform(
    t: np.ndarray, hr_bpm: float, rng: np.random.Generator, hr_std_bpm: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p(t) zero-mean unit-std, instantaneous HR in bpm)."""
    dt = np.diff(t, prepend=t[0])
    # slow OU-like HR drift
    hr = np.empty_like(t)
    x = 0.0
    for i, d in enumerate(dt):
        x += -x * d / 10.0 + hr_std_bpm * np.sqrt(2 * d / 10.0) * rng.standard_normal()
        hr[i] = hr_bpm + x
    phase = 2 * np.pi * np.cumsum(hr / 60.0 * dt)
    p = np.sin(phase) + 0.45 * np.sin(2 * phase - 0.9) + 0.15 * np.sin(3 * phase - 1.7)
    return (p - p.mean()) / p.std(), hr


def synthetic_bundle(
    duration_s: float = 30.0,
    fs: float = 30.0,
    hr_bpm: float = 72.0,
    skin: str | np.ndarray = "light",
    pulse_amp: float = 0.004,
    noise_a: float = 2e-4,
    noise_b: float = 1e-6,
    n_pix: int = 300,
    jitter_ms: float = 2.0,
    drop_frac: float = 0.0,
    illum_drift: float = 0.03,
    gt_fs: float = 60.0,
    lag_s: float = 0.15,
    gt_sign: int = 1,
    spec: PatchSpec | None = None,
    seed: int = 0,
    spo2: float | None = None,
    spo2_kappa: float = 0.04,
    spo2_ref: float = 97.0,
) -> TraceBundle:
    rng = np.random.default_rng(seed)
    spec = spec or PatchSpec()
    p_n = spec.num_patches
    n = int(duration_s * fs)
    t = np.arange(n) / fs + rng.normal(0, jitter_ms / 1000.0, n)
    t = np.sort(t - t[0])
    if drop_frac > 0:
        keep = rng.random(n) > drop_frac
        keep[[0, -1]] = True
        t = t[keep]
    # continuous-time pulse on a fine grid, sampled at frame times and at GT times
    t_fine = np.arange(-2.0, duration_s + 2.0, 1.0 / 240.0)
    p_fine, hr_fine = pulse_waveform(t_fine, hr_bpm, rng)
    p = np.interp(t, t_fine, p_fine)

    dc = SKIN_PRESETS[skin] if isinstance(skin, str) else np.asarray(skin, dtype=np.float64)
    region = spec.region_ids
    region_gain = np.array([1.0, 0.9, 0.6, 0.85, 0.85, 0.5])  # forehead/cheeks strongest
    amp = pulse_amp * region_gain[region] * rng.uniform(0.7, 1.3, p_n)
    shade = rng.uniform(0.85, 1.15, p_n)
    illum = 1.0 + illum_drift * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))

    pbv = PBV.copy()
    if spo2 is not None:
        pbv[0] *= 1.0 + spo2_kappa * (spo2_ref - spo2)
    dc_k = dc[None, :] * shade[:, None]  # [P, 3]
    clean = (
        dc_k[None]
        * illum[:, None, None]
        * (1.0 + amp[None, :, None] * pbv[None, None, :] / np.linalg.norm(pbv) * p[:, None, None])
    )
    var = (noise_a * clean + noise_b) / n_pix
    rgb = clean + rng.standard_normal(clean.shape) * np.sqrt(var)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    gt_t = np.arange(0.0, duration_s, 1.0 / gt_fs)
    gt = gt_sign * np.interp(gt_t - lag_s, t_fine, p_fine)  # finger pulse arrives lag_s later
    ita = ita_from_linear_rgb(rgb.mean(axis=0)).astype(np.float32)
    return TraceBundle(
        t=t,
        rgb_lin=rgb,
        rgb_srgb=rgb.copy(),
        valid=np.ones((t.size, p_n), bool),
        n_pix=np.full((t.size, p_n), n_pix, np.int32),
        center=np.zeros((t.size, p_n, 2), np.float32),
        motion=np.zeros(t.size, np.float32),
        face_found=np.ones(t.size, bool),
        ita_patch=ita,
        region_ids=region,
        gt_ppg=gt.astype(np.float32),
        gt_t=gt_t,
        gt_hr=np.interp(gt_t, t_fine, hr_fine).astype(np.float32),
        gt_spo2=None if spo2 is None else np.full(gt_t.size, np.round(spo2), np.float32),
        meta={
            "dataset": "synthetic",
            "subject": f"syn{seed}",
            "video_id": f"syn{seed}",
            "true_hr_bpm": hr_bpm,
            "true_lag_s": lag_s,
            "true_spo2": spo2,
            "skin": skin if isinstance(skin, str) else "custom",
        },
    )
