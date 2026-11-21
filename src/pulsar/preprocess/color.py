"""Colour-space utilities: sRGB <-> linear RGB, CIELAB, and the Individual Typology Angle (ITA).

All physics-based processing (PhysTone-Aug, IllumSim-IN, patch averaging) happens in *linear* RGB.
Camera tone curves are approximated by the sRGB transfer function.
"""

from __future__ import annotations

import numpy as np

# sRGB (D65) linear RGB -> XYZ
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB-encoded values in [0, 1] -> linear RGB in [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb8_to_linear_lut() -> np.ndarray:
    """256-entry LUT: uint8 sRGB -> float32 linear. Mirrors the on-device implementation."""
    return srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)


def linear_rgb_to_lab(rgb_lin: np.ndarray) -> np.ndarray:
    """Linear RGB [..., 3] -> CIELAB [..., 3] (D65)."""
    xyz = np.asarray(rgb_lin, dtype=np.float64) @ _M_RGB2XYZ.T
    t = xyz / _WHITE_D65
    d = 6.0 / 29.0
    f = np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def ita_degrees(lab: np.ndarray) -> np.ndarray:
    """Individual Typology Angle: atan2(L* - 50, b*) in degrees. Higher = lighter skin."""
    lab = np.asarray(lab, dtype=np.float64)
    return np.degrees(np.arctan2(lab[..., 0] - 50.0, lab[..., 2]))


def ita_from_linear_rgb(rgb_lin: np.ndarray) -> np.ndarray:
    return ita_degrees(linear_rgb_to_lab(rgb_lin))


# Chardon et al. (1991) descriptive bands (analysis itself uses ITA continuously).
ITA_BANDS: list[tuple[str, float, float]] = [
    ("very light", 55.0, 90.0),
    ("light", 41.0, 55.0),
    ("intermediate", 28.0, 41.0),
    ("tan", 10.0, 28.0),
    ("brown", -30.0, 10.0),
    ("dark", -90.0, -30.0),
]


def ita_band(ita: float) -> str:
    for name, lo, hi in ITA_BANDS:
        if lo < ita <= hi:
            return name
    return "dark" if ita <= -30 else "very light"
