"""Subject-level cluster bootstrap for confidence intervals (docs 06 §7).

Subjects (clusters) are resampled with replacement *within each group*, so group sizes are
preserved. Paired comparisons reuse the same resampled subject sets for both methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _resample_subjects(df: pd.DataFrame, rng: np.random.Generator, group_col: str | None) -> pd.DataFrame:
    parts = []
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=False))
    for _, gdf in groups:
        subs = gdf["subject"].unique()
        pick = rng.choice(subs, size=len(subs), replace=True)
        by = dict(tuple(gdf.groupby("subject", sort=False)))
        for k, s in enumerate(pick):
            part = by[s].copy()
            part["subject"] = f"{s}#{k}"  # keep duplicates distinct
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def cluster_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    group_col: str | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    point = float(stat(df))
    vals = np.array([stat(_resample_subjects(df, rng, group_col)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals.size else (np.nan, np.nan)
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": int(vals.size)}


def paired_bootstrap_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    group_col: str | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """CI of stat(A) - stat(B), resampling the *same* subjects for both (same windows assumed)."""
    rng = np.random.default_rng(seed)
    point = float(stat(df_a) - stat(df_b))
    subs_by_group = (
        {None: df_a["subject"].unique()}
        if group_col is None
        else {g: d["subject"].unique() for g, d in df_a.groupby(group_col)}
    )
    ga = dict(tuple(df_a.groupby("subject")))
    gb = dict(tuple(df_b.groupby("subject")))
    diffs = []
    for _ in range(n_boot):
        pa, pb = [], []
        for _, subs in subs_by_group.items():
            for k, s in enumerate(rng.choice(subs, size=len(subs), replace=True)):
                a, b = ga[s].copy(), gb[s].copy()
                a["subject"] = b["subject"] = f"{s}#{k}"
                pa.append(a)
                pb.append(b)
        diffs.append(stat(pd.concat(pa)) - stat(pd.concat(pb)))
    d = np.asarray(diffs)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": n_boot}
