"""TraceNet: per-patch temporal encoder + SNR-supervised patch attention + DFT HR head (docs 04 §1).

Small (~10^5 params), conv/BN/ReLU/matmul only -> INT8-friendly and LiteRT-exportable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulsar.models.features import N_FEATURES, TraceFeatures


@dataclass
class TraceNetConfig:
    width: int = 48
    enc_blocks: int = 4
    dec_blocks: int = 2
    kernel: int = 5
    fs: float = 30.0
    t_len: int = 300
    bpm_lo: float = 40.0
    bpm_hi: float = 200.0
    n_regions: int = 6
    adaptive: bool = True  # C3: reliability-weighted patch attention (False = uniform pooling)
    tau_init: float = 3.0


class DSBlock(nn.Module):
    """Residual depthwise-separable temporal conv block."""

    def __init__(self, c: int, k: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=dilation * (k // 2), dilation=dilation, groups=c, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)))
        return F.relu(x + self.bn2(self.pw(y)))


class DFTHead(nn.Module):
    """Power spectrum of the BVP at exact 1-bpm bins via a fixed (cos, sin) basis."""

    def __init__(self, t_len: int, fs: float, bpm: np.ndarray):
        super().__init__()
        t = np.arange(t_len) / fs
        f = bpm / 60.0
        win = np.hanning(t_len)
        self.register_buffer(
            "cos", torch.tensor(win[:, None] * np.cos(2 * np.pi * np.outer(t, f)), dtype=torch.float32)
        )
        self.register_buffer(
            "sin", torch.tensor(win[:, None] * np.sin(2 * np.pi * np.outer(t, f)), dtype=torch.float32)
        )
        self.log_scale = nn.Parameter(torch.zeros(()))  # learned inverse temperature (log)

    def forward(self, bvp: torch.Tensor) -> torch.Tensor:
        x = bvp - bvp.mean(-1, keepdim=True)
        p = (x @ self.cos) ** 2 + (x @ self.sin) ** 2  # [B, K]
        p = p / (p.sum(-1, keepdim=True) + 1e-8)
        return torch.exp(self.log_scale) * torch.log(p + 1e-6)


class TraceNet(nn.Module):
    def __init__(self, cfg: TraceNetConfig, region_ids: np.ndarray):
        super().__init__()
        self.cfg = cfg
        c = cfg.width
        self.features = TraceFeatures(region_ids, cfg.n_regions, cfg.fs)
        self.stem = nn.Sequential(
            nn.Conv1d(N_FEATURES, c, cfg.kernel, padding=cfg.kernel // 2, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(*[DSBlock(c, cfg.kernel, 2**i) for i in range(cfg.enc_blocks)])
        self.rel_head = nn.Sequential(
            nn.Linear(2 * c + self.features.n_desc, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(cfg.tau_init)))
        self.decoder = nn.Sequential(*[DSBlock(c, cfg.kernel, 2**i) for i in range(cfg.dec_blocks)])
        self.out = nn.Conv1d(c, 1, 1)
        self.register_buffer(
            "bpm", torch.tensor(np.arange(cfg.bpm_lo, cfg.bpm_hi + 0.5, 1.0), dtype=torch.float32)
        )
        self.hr_head = DFTHead(cfg.t_len, cfg.fs, self.bpm.numpy())

    def forward(self, x, valid, n_pix, motion) -> dict[str, torch.Tensor]:
        """x [B, P, T, 3] linear RGB; valid [B, P, T] bool; n_pix [B, P]; motion [B]."""
        feats, desc, mask, pos_raw = self.features(x, valid, n_pix, motion)
        b, p, f, t = feats.shape
        h = self.encoder(self.stem(feats.reshape(b * p, f, t)))  # [BP, C, T]
        c = h.shape[1]
        stats = torch.cat([h.mean(-1), h.std(-1)], dim=-1).reshape(b, p, 2 * c)
        rel = self.rel_head(torch.cat([stats, desc], dim=-1)).squeeze(-1)  # predicted SNR / 10 dB
        logits = rel / torch.exp(self.log_tau) if self.cfg.adaptive else torch.zeros_like(rel)
        logits = logits.masked_fill(~mask, -1e4)
        alpha = torch.softmax(logits, dim=1)  # [B, P]
        h4 = h.reshape(b, p, c, t)
        h4 = torch.where(mask.reshape(b, p, 1, 1), h4, torch.zeros_like(h4))  # 0 * NaN stays NaN; select
        z = (alpha.reshape(b, p, 1, 1) * h4).sum(1)
        bvp = self.out(self.decoder(z)).squeeze(1)  # [B, T]
        return {
            "bvp": bvp,
            "hr_logits": self.hr_head(bvp),
            "rel": rel,
            "alpha": alpha,
            "patch_mask": mask,
            "pos_raw": pos_raw,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def hr_from_logits(logits: torch.Tensor, bpm: torch.Tensor, temperature: float = 1.0, tol_bpm: float = 5.0):
    """-> (soft-argmax HR [B], mode HR [B], confidence [B] = prob mass within +-tol of the mode)."""
    q = torch.softmax(logits / temperature, dim=-1)
    hr_soft = (q * bpm).sum(-1)
    mode = bpm[q.argmax(-1)]
    near = (bpm.unsqueeze(0) - mode.unsqueeze(-1)).abs() <= tol_bpm
    conf = (q * near).sum(-1)
    return hr_soft, mode, conf
