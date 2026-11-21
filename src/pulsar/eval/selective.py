"""Selective prediction (abstention) and calibration metrics -- contribution C4 (docs 04 §7, 06 §4).

A model that is 'fair' only because it abstains more often on darker skin is not fair: every
selective result is reported per group together with coverage, and summarised by the coverage
parity ratio CPR = min_g coverage_g / max_g coverage_g at one *global* threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_softmax


def risk_coverage(err: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk (mean |error|) when keeping the k most confident predictions."""
    err = np.abs(np.asarray(err, float))
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    e = err[order]
    k = np.arange(1, e.size + 1)
    return k / e.size, np.cumsum(e) / k


def aurc(err: np.ndarray, conf: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = better ranking of errors by confidence)."""
    cov, risk = risk_coverage(err, conf)
    return float(np.trapezoid(risk, cov)) if cov.size > 1 else float(risk.mean())


def threshold_for_target_risk(err: np.ndarray, conf: np.ndarray, target_mae: float) -> float:
    """Lowest confidence threshold whose selective MAE is <= target (max coverage)."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(conf, float)[order]
    cov, risk = risk_coverage(err, conf)
    ok = np.nonzero(risk <= target_mae)[0]
    return float(c[ok[-1]]) if ok.size else float("inf")


def selective_by_group(
    df: pd.DataFrame, group_col: str, tau: float, err_col: str = "abs_err", conf_col: str = "conf"
) -> pd.DataFrame:
    """Per-group coverage and selective MAE at a global threshold tau (subject-level averages)."""
    rows = []
    for g, d in df.groupby(group_col):
        keep = d[conf_col] >= tau
        per_subj_cov = keep.groupby(d["subject"]).mean()
        kept = d[keep]
        sel_mae = kept.groupby("subject")[err_col].mean().mean() if len(kept) else np.nan
        rows.append(
            {
                group_col: g,
                "coverage": float(per_subj_cov.mean()),
                "selective_mae": float(sel_mae),
                "aurc": aurc(d[err_col].to_numpy(), d[conf_col].to_numpy()),
                "n_subjects": int(d["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).set_index(group_col)


def coverage_parity_ratio(table: pd.DataFrame) -> float:
    cov = table["coverage"]
    return float(cov.min() / cov.max()) if cov.max() > 0 else float("nan")


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for a binary event (e.g. |HR error| <= 5 bpm)."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def confidence_from_logits(
    logits: np.ndarray, bpm: np.ndarray, temperature: float = 1.0, tol_bpm: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """(mode HR, prob mass within +-tol of the mode) -- numpy twin of models.tracenet.hr_from_logits."""
    q = np.exp(log_softmax(np.asarray(logits, float) / temperature, axis=-1))
    mode = bpm[q.argmax(-1)]
    near = np.abs(bpm[None, :] - mode[:, None]) <= tol_bpm
    return mode, (q * near).sum(-1)


def fit_temperature(
    logits: np.ndarray,
    hr_true: np.ndarray,
    bpm: np.ndarray,
    sigma: float = 1.5,
    grid: np.ndarray | None = None,
) -> float:
    """Temperature minimising the soft-target NLL on a validation set (1-D grid search, robust)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121)) if grid is None else grid
    target = np.exp(-((bpm[None, :] - np.asarray(hr_true, float)[:, None]) ** 2) / (2 * sigma**2))
    target /= target.sum(-1, keepdims=True)
    nll = [-(target * log_softmax(np.asarray(logits, float) / t, axis=-1)).sum(-1).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])
