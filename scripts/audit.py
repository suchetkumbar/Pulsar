"""M2 fairness & robustness audit of training-free baselines (contribution C0).

    uv run python scripts/audit.py --cache data/cache/traces/mmpd --group fitzpatrick
    uv run python scripts/audit.py --synthetic            # pipeline check without datasets

For every window: apply IllumSim-IN at severities S0..S4 (seeded per window, identical across
methods), run each method on the pixel-weighted face mean, estimate HR on the exact bpm grid.
Writes predictions (parquet), markdown tables with subject-bootstrap CIs, and figures to
reports/<name>/.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from pulsar.aug.illumsim import severity_preset, simulate
from pulsar.data.loading import load_cache_windows, synthetic_windows
from pulsar.eval.metrics import ita_group
from pulsar.eval.report import df_markdown, group_table, plot_group_bars, plot_severity, to_markdown
from pulsar.signal.classical import METHODS, run_method
from pulsar.signal.filters import postprocess_bvp
from pulsar.signal.spectral import hr_grid

FS = 30.0


def face_mean(x: np.ndarray, valid: np.ndarray, n_pix: np.ndarray) -> np.ndarray:
    keep = valid.mean(axis=0) >= 0.8
    w = valid * n_pix[None, :] * keep[None, :]
    den = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w[..., None] * x).sum(axis=1) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*", type=Path, default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["POS", "CHROM", "GREEN", "LGI"], choices=sorted(METHODS))
    ap.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--group", default=None, help="group column (default: fitzpatrick if present, else ITA bins)"
    )
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--name", default="m2_audit")
    ap.add_argument(
        "--from-predictions", action="store_true", help="re-use outputs/<name>_predictions.parquet"
    )
    args = ap.parse_args()

    pred_path = Path("outputs") / f"{args.name}_predictions.parquet"
    if args.from_predictions:
        report(pd.read_parquet(pred_path), args, pred_path)
        return
    ws = (
        synthetic_windows(n_subjects=18, stride_s=args.stride)
        if args.synthetic
        else load_cache_windows(args.cache, FS, 10.0, args.stride)
    )
    rows = []
    for w in ws:
        meta = {
            "dataset": w.meta.get("dataset"),
            "subject": f"{w.meta.get('dataset')}:{w.subject}",
            "video_id": w.video_id,
            "ita": w.meta.get("ita"),
            "fitzpatrick": w.meta.get("fitzpatrick"),
            "light": w.meta.get("light"),
            "ita_group": ita_group(w.meta.get("ita")),
        }
        for j in range(len(w)):
            for s in args.severities:
                rng = np.random.default_rng(zlib.crc32(f"{w.video_id}|{j}|{s}".encode()))
                x, _ = simulate(w.x[j], w.center[j], w.n_pix[j], FS, rng, severity_preset(s))
                rgb = face_mean(x, w.valid[j], w.n_pix[j])
                if not np.isfinite(rgb).all():
                    continue
                for m in args.methods:
                    hr = hr_grid(postprocess_bvp(run_method(m, rgb, FS), FS), FS)
                    rows.append(
                        meta
                        | {
                            "method": m,
                            "severity": s,
                            "start_s": float(w.start_s[j]),
                            "hr_true": float(w.hr[j]),
                            "hr_pred": hr,
                        }
                    )
    df = pd.DataFrame(rows)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pred_path)
    report(df, args, pred_path)


def report(df: pd.DataFrame, args: argparse.Namespace, pred_path: Path) -> None:
    group = args.group or ("fitzpatrick" if df["fitzpatrick"].notna().any() else "ita_group")
    df = df[df[group].notna()]
    # synthetic runs are pipeline checks, never results -> keep them out of the committed reports/
    out = (Path("outputs") if args.synthetic else Path("reports")) / args.name
    out.mkdir(parents=True, exist_ok=True)

    s0 = df[df["severity"] == 0]
    t = group_table(s0, group, ["method"], n_boot=args.n_boot)
    t.to_csv(out / "fairness_S0.csv", index=False)
    worst = t[t[group] == "WORST"][["method", "mae", "lo", "hi", "worst_group", "gap"]]
    md = [
        f"# {args.name}\n",
        f"Windows: {len(s0) // len(args.methods)}, subjects: {s0['subject'].nunique()}, "
        f"group column: `{group}`\n",
        "## Subject-level HR MAE by group (S0, 95% subject-bootstrap CI)\n",
        to_markdown(t, ["method"], group),
        "\n\n## Worst group\n",
        df_markdown(worst, index=False),
    ]
    sev = df.groupby(["method", "severity"]).apply(
        lambda d: (
            d.groupby("subject")
            .apply(lambda x: (x["hr_pred"] - x["hr_true"]).abs().mean(), include_groups=False)
            .mean()
        ),
        include_groups=False,
    )
    md += ["\n\n## Robustness: subject-level MAE vs IllumSim-IN severity\n", df_markdown(sev.unstack())]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    plot_group_bars(t, group, "method", out / "mae_by_group_S0.png", f"Baselines by {group} (S0)")
    plot_severity(df, "method", out / "mae_vs_severity.png", "Robustness to Indian indoor lighting")
    print("\n".join(md))
    print(f"\nwrote {out}/ and {pred_path}")


if __name__ == "__main__":
    main()
