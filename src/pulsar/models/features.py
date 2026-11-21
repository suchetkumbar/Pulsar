"""In-graph feature builder for TraceNet (docs 04 §1.1).

Everything here uses only conv1d / matmul / elementwise ops so the whole pre-processing after
resampling exports to LiteRT/ONNX unchanged -- Python, desktop and Android cannot diverge.

Input  x [B, P, T, 3] linear RGB, valid [B, P, T], n_pix [B, P], motion [B]
Output feats [B, P, F=5, T] (z-scored per patch), desc [B, P, D], patch_mask [B, P], pos_raw
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

_EPS = 1e-6
_M_RGB2XYZ = torch.tensor(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
N_FEATURES = 5


def same_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise 'same' conv with zero padding. x [N, C, T], kernel [K] (odd)."""
    c = x.shape[1]
    w = kernel.view(1, 1, -1).repeat(c, 1, 1)
    return F.conv1d(x, w, padding=kernel.numel() // 2, groups=c)


class MovingAverage(nn.Module):
    """Edge-normalised moving average (zero-pad conv divided by conv of ones)."""

    def __init__(self, length: int):
        super().__init__()
        length += 1 - length % 2  # odd
        self.register_buffer("w", torch.full((length,), 1.0 / length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num = same_conv(x, self.w)
        den = same_conv(torch.ones_like(x[:, :1]), self.w)
        return num / den


class FIRBandpass(nn.Module):
    def __init__(self, fs: float, band: tuple[float, float], numtaps: int = 61):
        super().__init__()
        taps = firwin(numtaps, list(band), pass_zero=False, fs=fs, window="hamming")
        self.register_buffer("w", torch.tensor(taps, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return same_conv(x, self.w)


def std_ratio(num_sq: torch.Tensor, den_sq: torch.Tensor) -> torch.Tensor:
    """sqrt(num/den) of two moving second moments, exact, with 0/0 (an all-invalid patch) -> 0.

    No ``+ eps**2`` guard: the onnxscript optimizer folds ``x + 1e-12`` into ``x`` (constant treated as
    zero), which turned all-invalid patches into NaN in dynamo-exported graphs. A larger eps would bias
    the ratio most where pulse power is smallest (darker skin), so the guard is a select instead.
    """
    den = torch.where(den_sq > 0, den_sq, torch.ones_like(den_sq))
    return torch.sqrt(num_sq.clamp(min=0.0) / den)


def lab_ita(rgb_lin: torch.Tensor) -> torch.Tensor:
    """ITA (deg) from linear RGB [..., 3]."""
    xyz = rgb_lin @ _M_RGB2XYZ.to(rgb_lin).T
    t = xyz / _WHITE.to(rgb_lin)
    d = 6.0 / 29.0
    f = torch.where(t > d**3, torch.clamp(t, min=1e-9) ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)
    lightness = 116.0 * f[..., 1] - 16.0
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.atan2(lightness - 50.0, b) * (180.0 / math.pi)  # rad2deg has no ONNX symbol


class TraceFeatures(nn.Module):
    def __init__(
        self,
        region_ids: np.ndarray,
        n_regions: int = 6,
        fs: float = 30.0,
        band: tuple[float, float] = (0.7, 3.5),
        norm_win_s: float = 1.6,
        min_valid_frac: float = 0.8,
    ):
        super().__init__()
        self.ma = MovingAverage(int(round(norm_win_s * fs)))
        self.bp = FIRBandpass(fs, band)
        onehot = F.one_hot(torch.as_tensor(region_ids, dtype=torch.long), n_regions).float()
        self.register_buffer("region_onehot", onehot)  # [P, R]
        self.min_valid_frac = min_valid_frac
        self.n_desc = 6 + n_regions

    def _fill_invalid(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vf = v.unsqueeze(-1).float()
        mean = (x * vf).sum(2, keepdim=True) / vf.sum(2, keepdim=True).clamp(min=1.0)
        return torch.where(v.unsqueeze(-1), x, mean.expand_as(x))

    def forward(self, x, valid, n_pix, motion):
        b, p, t, _ = x.shape
        x = self._fill_invalid(x, valid)
        xc = x.reshape(b * p, t, 3).transpose(1, 2)  # [BP, 3, T]
        mu = self.ma(xc)
        cn = xc / (mu + _EPS)  # ~ 1 + pulse
        # POS (moving-window form): S1 = G - B, S2 = -2R + G + B, h = S1 + (std1/std2) S2
        s1 = cn[:, 1:2] - cn[:, 2:3]
        s2 = -2 * cn[:, 0:1] + cn[:, 1:2] + cn[:, 2:3]
        s1 = s1 - self.ma(s1)
        s2 = s2 - self.ma(s2)
        pos = self.bp(s1 + std_ratio(self.ma(s1 * s1), self.ma(s2 * s2)) * s2)
        # CHROM (moving-window form)
        xs = self.bp(3 * cn[:, 0:1] - 2 * cn[:, 1:2])
        ys = self.bp(1.5 * cn[:, 0:1] + cn[:, 1:2] - 1.5 * cn[:, 2:3])
        chrom = xs - std_ratio(self.ma(xs * xs), self.ma(ys * ys)) * ys
        rgb = self.bp(cn - 1.0)
        raw = torch.cat([rgb, pos, chrom], dim=1)  # [BP, 5, T]
        std = raw.std(dim=2, keepdim=True)
        feats = (raw / (std + _EPS)).reshape(b, p, N_FEATURES, t)

        vfrac = valid.float().mean(-1)  # [B, P]
        patch_mask = vfrac >= self.min_valid_frac
        mean_rgb = x.mean(2)  # [B, P, 3]
        lum = (mean_rgb * torch.tensor([0.2126, 0.7152, 0.0722], device=x.device)).sum(-1)
        desc = torch.cat(
            [
                torch.log(lum + 1e-4).unsqueeze(-1) / 5.0,
                (lab_ita(mean_rgb) / 50.0).unsqueeze(-1),
                (torch.log1p(n_pix.float()) / 6.0).unsqueeze(-1),
                vfrac.unsqueeze(-1),
                motion.float().view(b, 1, 1).expand(b, p, 1) * 10.0,
                (torch.log(std[:, 3:4, 0].reshape(b, p, 1) + 1e-6) / 10.0),
                self.region_onehot.unsqueeze(0).expand(b, p, -1),
            ],
            dim=-1,
        )
        pos_raw = raw[:, 3].reshape(b, p, t)
        return feats, desc, patch_mask, pos_raw
