"""Per-face adaptive skin masks.

Fixed YCrCb/HSV skin thresholds are tuned on lighter skin and silently discard pixels on darker
faces (a fairness bug). Instead we fit a Gaussian to the chromaticity of seed pixels from this
face's own cheek patches and keep pixels within a Mahalanobis radius. Thresholds on luminance are
*relative* to this face's median, never absolute.
"""

from __future__ import annotations

import numpy as np

CHI2_2DOF_99 = 9.21  # 99% quantile of chi^2 with 2 dof


def chroma_rg(rgb: np.ndarray) -> np.ndarray:
    """Normalised rg chromaticity [..., 2] from linear RGB [..., 3] (intensity-invariant)."""
    s = rgb.sum(axis=-1, keepdims=True) + 1e-9
    return (rgb / s)[..., :2]


class SkinModel:
    """Gaussian skin-chroma model with a running update across frames."""

    def __init__(self, momentum: float = 0.95, radius2: float = CHI2_2DOF_99):
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.lum_median: float | None = None
        self.momentum = momentum
        self.radius2 = radius2

    def update(self, seed_rgb_lin: np.ndarray) -> None:
        """Update from seed pixels [N, 3] (linear RGB)."""
        if seed_rgb_lin.shape[0] < 20:
            return
        c = chroma_rg(seed_rgb_lin)
        mu = c.mean(axis=0)
        cov = np.cov(c, rowvar=False) + np.eye(2) * 1e-6
        lum = float(np.median(seed_rgb_lin.mean(axis=1)))
        if self.mean is None:
            self.mean, self.cov, self.lum_median = mu, cov, lum
        else:
            m = self.momentum
            self.mean = m * self.mean + (1 - m) * mu
            self.cov = m * self.cov + (1 - m) * cov
            self.lum_median = m * self.lum_median + (1 - m) * lum

    def mask(self, rgb_lin: np.ndarray) -> np.ndarray:
        """Boolean skin mask for pixels [..., 3]. All-True until the model is initialised."""
        if self.mean is None:
            return np.ones(rgb_lin.shape[:-1], dtype=bool)
        d = chroma_rg(rgb_lin) - self.mean
        inv = np.linalg.inv(self.cov)
        m2 = np.einsum("...i,ij,...j->...", d, inv, d)
        lum = rgb_lin.mean(axis=-1)
        not_shadow = lum > 0.35 * self.lum_median  # hair / deep shadow, relative to this face
        return (m2 < self.radius2) & not_shadow


def specular_mask(rgb_lin: np.ndarray, pct: float = 98.0, sat_max: float = 0.15) -> np.ndarray:
    """True for pixels that are NOT specular highlights (bright *and* desaturated)."""
    lum = rgb_lin.mean(axis=-1)
    if lum.size == 0:
        return np.ones(lum.shape, dtype=bool)
    mx = rgb_lin.max(axis=-1)
    mn = rgb_lin.min(axis=-1)
    sat = (mx - mn) / (mx + 1e-9)
    thr = np.percentile(lum, pct)
    return ~((lum >= thr) & (sat < sat_max))
