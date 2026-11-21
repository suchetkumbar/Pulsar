"""Fairness tables (with subject-bootstrap CIs) and standard figures (docs 06 §8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pulsar.eval.bootstrap import cluster_bootstrap
from pulsar.eval.metrics import subject_level


def group_table(
    df: pd.DataFrame, group_col: str, by: list[str], n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Subject-level MAE per (by..., group) with 95% bootstrap CI, plus worst-group and gap rows."""

    def mae(d: pd.DataFrame) -> float:
        return float(subject_level(d)["mae"].iloc[0])

    rows = []
    for keys, d in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys, strict=True))
        for g, dg in d.groupby(group_col):
            ci = cluster_bootstrap(dg, mae, n_boot=n_boot, seed=seed)
            rows.append(
                base
                | {
                    group_col: g,
                    "mae": ci["point"],
                    "lo": ci["lo"],
                    "hi": ci["hi"],
                    "n_subjects": dg["subject"].nunique(),
                    "n_windows": len(dg),
                }
            )
        overall = cluster_bootstrap(d, mae, group_col=group_col, n_boot=n_boot, seed=seed)
        g_mae = subject_level(d, group_col)["mae"]

        def worst(x: pd.DataFrame) -> float:
            return float(subject_level(x, group_col)["mae"].max())

        w = cluster_bootstrap(d, worst, group_col=group_col, n_boot=n_boot, seed=seed)
        rows.append(
            base
            | {
                group_col: "ALL",
                "mae": overall["point"],
                "lo": overall["lo"],
                "hi": overall["hi"],
                "n_subjects": d["subject"].nunique(),
                "n_windows": len(d),
            }
        )
        rows.append(
            base
            | {
                group_col: "WORST",
                "mae": w["point"],
                "lo": w["lo"],
                "hi": w["hi"],
                "worst_group": str(g_mae.idxmax()),
                "gap": float(g_mae.max() - g_mae.min()),
            }
        )
    return pd.DataFrame(rows)


def df_markdown(df: pd.DataFrame, index: bool = True, floatfmt: str = ".2f") -> str:
    """Minimal GitHub-markdown table (avoids the optional `tabulate` dependency)."""
    d = df.reset_index() if index else df

    def fmt(v) -> str:
        if isinstance(v, float | np.floating):
            return "" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def to_markdown(t: pd.DataFrame, by: list[str], group_col: str) -> str:
    t = t.copy()
    t["MAE [95% CI]"] = t.apply(lambda r: f"{r['mae']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]", axis=1)
    t[group_col] = t[group_col].astype(str)
    wide = t.pivot_table(index=by, columns=group_col, values="MAE [95% CI]", aggfunc="first")
    return df_markdown(wide)


def plot_group_bars(
    t: pd.DataFrame, group_col: str, method_col: str, out: str | Path, title: str = ""
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = t[~t[group_col].isin(["ALL", "WORST"])]
    groups = sorted(d[group_col].astype(str).unique())
    methods = list(dict.fromkeys(d[method_col]))
    x = np.arange(len(groups))
    wbar = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(groups), 3.6))
    for i, m in enumerate(methods):
        dm = d[d[method_col] == m]
        dm = dm.set_index(dm[group_col].astype(str)).reindex(groups)
        err = np.vstack([dm["mae"] - dm["lo"], dm["hi"] - dm["mae"]])
        ax.bar(x + (i - (len(methods) - 1) / 2) * wbar, dm["mae"], wbar, yerr=err, capsize=2, label=m)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Subject-level HR MAE (bpm)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_severity(df: pd.DataFrame, method_col: str, out: str | Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m, d in df.groupby(method_col):
        s = d.groupby("severity").apply(lambda x: subject_level(x)["mae"].iloc[0], include_groups=False)
        ax.plot(s.index, s.values, marker="o", label=m)
    sev = sorted(df["severity"].unique())
    ax.set_xticks(sev, [f"S{int(v)}" for v in sev])
    ax.set_xlabel("IllumSim-IN severity")
    ax.set_ylabel("Subject-level HR MAE (bpm)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
