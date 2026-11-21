"""Poisson-Gaussian camera noise for patch-mean traces.

Per pixel: Var = a*I + b (shot + read noise, linear units). A patch mean over N pixels has
variance (a*I + b)/N. When an augmentation changes the true intensity from I to I' = g*I + c,
the existing noise is scaled by g, so the *additional* noise needed is
    delta^2 = [(a*I' + b) - g^2 (a*I + b)] / N   (clipped at 0 -- noise can't be removed).
"""

from __future__ import annotations

import numpy as np

DEFAULT_A = 2e-3  # shot-noise gain (linear-intensity units); calibrated per dataset in M3
DEFAULT_B = 2e-6  # read-noise floor


def topup_noise(
    x_new: np.ndarray,
    x_old: np.ndarray,
    gain: np.ndarray | float,
    n_pix: np.ndarray,
    rng: np.random.Generator,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
) -> np.ndarray:
    """Add the extra noise implied by an intensity change. Shapes broadcast; n_pix is [P]."""
    n = np.maximum(np.asarray(n_pix, dtype=np.float64), 1.0)[None, :, None]
    g = np.asarray(gain, dtype=np.float64)
    var_new = a * np.clip(x_new, 0, None) + b
    var_old_scaled = g**2 * (a * np.clip(x_old, 0, None) + b)
    extra = np.clip(var_new - var_old_scaled, 0, None) / n
    return x_new + rng.standard_normal(x_new.shape) * np.sqrt(extra)
