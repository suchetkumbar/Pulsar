"""HR error metrics and fairness aggregations (docs 06 §2, §4).

Subject is the independent unit: metrics are computed per subject first, then averaged
(`subject_level`). Window-pooled numbers are available but must be labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hr_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[ok], true[ok]
    if pred.size == 0:
        return {k: float("nan") for k in ("mae", "rmse", "mape", "pearson", "acc5")} | {"n": 0}
    err = pred - true
    r = float(np.corrcoef(pred, true)[0, 1]) if pred.size > 2 and pred.std() > 0 else float("nan")
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float((np.abs(err) / true).mean() * 100),
        "pearson": r,
        "acc5": float((np.abs(err) <= 5).mean()),
        "n": int(pred.size),
    }


def bland_altman(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    d = np.asarray(pred, float) - np.asarray(true, float)
    d = d[np.isfinite(d)]
    bias, sd = float(d.mean()), float(d.std(ddof=1)) if d.size > 1 else float("nan")
    return {"bias": bias, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd}


def subject_mae(df: pd.DataFrame, pred: str = "hr_pred", true: str = "hr_true") -> pd.Series:
    """Per-subject MAE from a per-window table with a 'subject' column."""
    e = (df[pred] - df[true]).abs()
    return e.groupby(df["subject"]).mean()


def subject_level(
    df: pd.DataFrame, group_col: str | None = None, pred: str = "hr_pred", true: str = "hr_true"
) -> pd.DataFrame:
    """Subject-level MAE averaged within groups (or overall). Columns: mae, n_subjects, n_windows."""
    d = df.dropna(subset=[pred, true]).copy()
    d["_ae"] = (d[pred] - d[true]).abs()
    keys = ["subject"] if group_col is None else [group_col, "subject"]
    per_subj = d.groupby(keys)["_ae"].agg(["mean", "size"]).reset_index()
    if group_col is None:
        return pd.DataFrame(
            {
                "mae": [per_subj["mean"].mean()],
                "n_subjects": [len(per_subj)],
                "n_windows": [int(per_subj["size"].sum())],
            }
        )
    out = per_subj.groupby(group_col).agg(
        mae=("mean", "mean"), n_subjects=("mean", "size"), n_windows=("size", "sum")
    )
    return out


def fairness_summary(
    df: pd.DataFrame, group_col: str, pred: str = "hr_pred", true: str = "hr_true"
) -> dict[str, float]:
    g = subject_level(df, group_col, pred, true)["mae"]
    overall = float(subject_level(df, None, pred, true)["mae"].iloc[0])
    return {
        "overall_mae": overall,
        "worst_group_mae": float(g.max()),
        "best_group_mae": float(g.min()),
        "gap": float(g.max() - g.min()),
        "ratio": float(g.max() / g.min()) if g.min() > 0 else float("nan"),
        "worst_group": str(g.idxmax()),
    }


def ita_slope(df: pd.DataFrame, ita_col: str = "ita", pred: str = "hr_pred", true: str = "hr_true") -> float:
    """OLS slope of subject-level MAE vs subject ITA (bpm per degree). Negative = worse when darker."""
    d = df.dropna(subset=[pred, true, ita_col]).copy()
    d["_ae"] = (d[pred] - d[true]).abs()
    s = d.groupby("subject").agg(mae=("_ae", "mean"), ita=(ita_col, "mean"))
    if len(s) < 3:
        return float("nan")
    return float(np.polyfit(s["ita"], s["mae"], 1)[0])


def ita_group(ita: float | None) -> str:
    """Coarse ITA bins used when no Fitzpatrick labels exist (light > 41 deg, mid 10-41, dark < 10)."""
    if ita is None or not np.isfinite(ita):
        return "unknown"
    return "light(>41)" if ita > 41 else ("mid(10-41)" if ita > 10 else "dark(<10)")
