"""TraceNet losses (docs 04 §1.3)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def neg_pearson_lagged(pred: torch.Tensor, target: torch.Tensor, max_shift: int = 6) -> torch.Tensor:
    """1 - max_{|s|<=max_shift} corr(pred[t], target[t+s]); tolerates residual GT lag (±0.2 s)."""
    t = pred.shape[-1]
    best = None
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            c = _corr(pred[..., : t - s], target[..., s:])
        else:
            c = _corr(pred[..., -s:], target[..., : t + s])
        best = c if best is None else torch.maximum(best, c)
    return (1 - best).mean()


def hr_soft_ce(logits: torch.Tensor, hr: torch.Tensor, bpm: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    target = torch.exp(-((bpm.unsqueeze(0) - hr.unsqueeze(-1)) ** 2) / (2 * sigma**2))
    target = target / target.sum(-1, keepdim=True)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


@torch.no_grad()
def patch_snr_db(
    sig: torch.Tensor,
    hr_bpm: torch.Tensor,
    fs: float,
    band=(0.7, 3.5),
    dev_hz: float = 0.1,
    step_hz: float = 0.02,
) -> torch.Tensor:
    """Per-patch SNR (dB) of signals sig [B, P, T] w.r.t. the GT HR [B] (de Haan-style)."""
    t_len = sig.shape[-1]
    f = torch.arange(band[0], band[1] + step_hz / 2, step_hz, device=sig.device)
    t = torch.arange(t_len, device=sig.device) / fs
    win = torch.hann_window(t_len, periodic=False, device=sig.device)
    arg = 2 * math.pi * t.unsqueeze(1) * f.unsqueeze(0)  # [T, K]
    x = (sig - sig.mean(-1, keepdim=True)) * win
    pw = (x @ torch.cos(arg)) ** 2 + (x @ torch.sin(arg)) ** 2  # [B, P, K]
    f0 = (hr_bpm / 60.0).view(-1, 1, 1)
    m = ((f.view(1, 1, -1) - f0).abs() <= dev_hz) | ((f.view(1, 1, -1) - 2 * f0).abs() <= dev_hz)
    s = (pw * m).sum(-1)
    n = (pw * (~m)).sum(-1)
    return 10 * torch.log10((s + 1e-12) / (n + 1e-12))


def reliability_loss(rel: torch.Tensor, snr_db: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Huber between predicted reliability (SNR/10 dB) and the measured per-patch SNR."""
    target = (snr_db / 10.0).clamp(-3, 3)
    loss = F.smooth_l1_loss(rel, target, reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def total_loss(
    out: dict,
    batch: dict,
    bpm: torch.Tensor,
    fs: float,
    w_pearson=1.0,
    w_ce=1.0,
    w_rel=0.1,
    supervise_rel: bool = True,
) -> tuple[torch.Tensor, dict]:
    lp = neg_pearson_lagged(out["bvp"], batch["bvp"])
    lc = hr_soft_ce(out["hr_logits"], batch["hr"], bpm)
    loss = w_pearson * lp + w_ce * lc
    logs = {"l_pearson": float(lp), "l_ce": float(lc)}
    if supervise_rel and w_rel > 0:
        snr = patch_snr_db(out["pos_raw"].detach().float(), batch["hr"], fs)
        lr = reliability_loss(out["rel"], snr, out["patch_mask"])
        loss = loss + w_rel * lr
        logs["l_rel"] = float(lr)
    return loss, logs
