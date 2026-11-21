"""PhysTone-Aug: physics-based, SNR-realistic skin-tone augmentation (docs 04 §3).

Model (linear RGB, per channel c): I_c = I^s_c + I^d_c, where only the diffuse (dermal) part
carries the pulse and is attenuated by melanin (Beer-Lambert in optical density):
    I^d_c -> k_c I^d_c,   k_c = exp(-sigma_m,c * delta)
With the surface fraction s_c = I^s_c / DC_c, the transform is affine in I and therefore commutes
with patch averaging (exact at trace level):
    c'_c(t) = k_c c_c(t) + (1 - k_c) s_c DC_c
Relative pulse amplitude drops by k/(k + (1-k)s), and the sensor-noise top-up adds the extra shot
/read noise of the dimmer signal. Naive colour-only augmentation (s = 0, no top-up) is available
for the A7 ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar.aug.sensor_noise import DEFAULT_A, DEFAULT_B, topup_noise
from pulsar.preprocess.color import ita_from_linear_rgb

# Effective melanin optical-density vector in camera R,G,B bands (relative). Placeholder that maps
# typical light -> dark skin chromaticity; replaced in M3 by a Tsumura-style ICA estimate fitted to
# training-set skin pixels (docs 04 §3.5). Ordering sigma_B > sigma_G > sigma_R is physical.
DEFAULT_SIGMA_MELANIN = np.array([1.0, 1.15, 1.30])


@dataclass(frozen=True)
class PhysToneConfig:
    p_apply: float = 0.5
    sigma_m: tuple[float, float, float] = tuple(DEFAULT_SIGMA_MELANIN)
    surface_frac: float = 0.10  # calibrated in M3 (search 0.02-0.25)
    ita_min: float = -40.0  # darkest target ITA
    noise_a: float = DEFAULT_A
    noise_b: float = DEFAULT_B
    noise_topup: bool = True  # False + surface_frac=0 -> "colour-only" ablation (A7)
    max_delta: float = 4.0


def darken(
    x: np.ndarray,
    delta: float,
    sigma_m: np.ndarray,
    surface_frac: float,
    dc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free melanin increase. x [T, P, 3]; returns (x', per-channel gain k [3])."""
    k = np.exp(-np.asarray(sigma_m, dtype=np.float64) * delta)
    dc = x.mean(axis=0, keepdims=True) if dc is None else dc
    xp = k * x + (1 - k) * surface_frac * dc
    return xp, k


def relative_pulse_factor(delta: float, sigma_m: np.ndarray, surface_frac: float) -> np.ndarray:
    """(AC/DC)' / (AC/DC) per channel = k / (k + (1-k) s)."""
    k = np.exp(-np.asarray(sigma_m, dtype=np.float64) * delta)
    return k / (k + (1 - k) * surface_frac)


def window_ita(x: np.ndarray, valid: np.ndarray | None = None) -> float:
    """ITA of the window's mean skin colour."""
    if valid is not None:
        m = valid.any(axis=0)
        x = x[:, m] if m.any() else x
    return float(ita_from_linear_rgb(np.nanmean(x.reshape(-1, 3), axis=0)))


def delta_for_target_ita(
    x: np.ndarray, target_ita: float, cfg: PhysToneConfig, valid: np.ndarray | None = None
) -> float:
    """Bisection for the melanin increase that brings the window's ITA to `target_ita`."""
    mean = np.nanmean(x.reshape(-1, 3), axis=0)
    sigma = np.asarray(cfg.sigma_m)

    def ita_at(d: float) -> float:
        k = np.exp(-sigma * d)
        return float(ita_from_linear_rgb(k * mean + (1 - k) * cfg.surface_frac * mean))

    if ita_at(0.0) <= target_ita:
        return 0.0
    lo, hi = 0.0, cfg.max_delta
    if ita_at(hi) > target_ita:
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ita_at(mid) > target_ita:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def augment(
    x: np.ndarray,
    n_pix: np.ndarray,
    rng: np.random.Generator,
    cfg: PhysToneConfig | None = None,
    target_ita: float | None = None,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Darken one window to a random (or given) target ITA. x: [T, P, 3] linear RGB."""
    cfg = cfg or PhysToneConfig()
    ita0 = window_ita(x, valid)
    if target_ita is None:
        target_ita = float(rng.uniform(cfg.ita_min, ita0)) if ita0 > cfg.ita_min else ita0
    delta = delta_for_target_ita(x, target_ita, cfg, valid)
    if delta <= 0:
        return x.copy(), {"delta": 0.0, "ita_from": ita0, "ita_to": ita0}
    xp, k = darken(x, delta, np.asarray(cfg.sigma_m), cfg.surface_frac)
    if cfg.noise_topup:
        xp = topup_noise(xp, x, k[None, None, :], n_pix, rng, cfg.noise_a, cfg.noise_b)
    return xp.astype(np.float32), {
        "delta": float(delta),
        "ita_from": ita0,
        "ita_to": window_ita(xp, valid),
        "k": k.tolist(),
    }


def naive_tone(x: np.ndarray, rng: np.random.Generator, strength: tuple[float, float] = (0.2, 0.8)):
    """Ablation A2: scale towards a fixed brown in sRGB-like fashion (no physics, no SNR change)."""
    brown = np.array([0.12, 0.065, 0.04])
    a = float(rng.uniform(*strength))
    mean = x.mean(axis=0, keepdims=True)
    return (x * ((1 - a) + a * brown / np.maximum(mean, 1e-6))).astype(np.float32)
