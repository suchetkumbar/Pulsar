"""Evaluate a trained TraceNet on held-out caches: fairness, robustness and abstention parity.

    uv run --no-sync python scripts/evaluate.py --model runs/<run>/best.pt \
        --cache data/cache/traces/mmpd --group fitzpatrick --name P4_A5_seed0

Outputs reports/<name>/ (markdown tables with subject-bootstrap CIs + figures) and
outputs/<name>_predictions.parquet. The abstention threshold --tau must be fixed in advance
(from validation), never tuned on the test set (docs 06 §6).
"""

from __future__ import annotations

import argparse
import copy
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from pulsar.aug.illumsim import severity_preset, simulate
from pulsar.data.loading import load_cache_windows, synthetic_windows
from pulsar.data.windows import Windows
from pulsar.eval.report import df_markdown, group_table, plot_group_bars, plot_severity, to_markdown
from pulsar.eval.selective import coverage_parity_ratio, ece, selective_by_group
from pulsar.train.dataset import WindowDataset
from pulsar.train.loop import load_model, predict, resolve_device

FS = 30.0


def with_severity(ws: list[Windows], level: int) -> list[Windows]:
    if level == 0:
        return ws
    out = []
    for w in ws:
        w2 = copy.copy(w)
        xs = np.empty_like(w.x)
        for j in range(len(w)):
            rng = np.random.default_rng(zlib.crc32(f"{w.video_id}|{j}|{level}".encode()))
            xs[j], _ = simulate(w.x[j], w.center[j], w.n_pix[j], FS, rng, severity_preset(level))
        w2.x = xs
        out.append(w2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", nargs="*", type=Path, default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--group", default=None)
    ap.add_argument("--severities", nargs="+", type=int, default=[0])
    ap.add_argument("--tau", type=float, default=0.5, help="global abstention threshold (fixed beforehand)")
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = load_model(args.model, device)
    ws = (
        synthetic_windows(n_subjects=12, stride_s=args.stride, seed=123)
        if args.synthetic
        else load_cache_windows(args.cache, FS, 10.0, args.stride)
    )
    frames = []
    for s in args.severities:
        df = predict(model, WindowDataset(with_severity(ws, s)), device)
        df["severity"] = s
        df["method"] = "TraceNet"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["abs_err"] = (df["hr_pred"] - df["hr_true"]).abs()
    group = args.group or ("fitzpatrick" if df["fitzpatrick"].notna().any() else "ita_group")
    df = df[df[group].notna()]

    out = (Path("outputs") if args.synthetic else Path("reports")) / args.name
    out.mkdir(parents=True, exist_ok=True)
    pred_path = Path("outputs") / f"{args.name}_predictions.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pred_path)

    s0 = df[df["severity"] == 0] if 0 in args.severities else df[df["severity"] == args.severities[0]]
    t = group_table(s0, group, ["method"], n_boot=args.n_boot)
    t.to_csv(out / "fairness.csv", index=False)
    sel = selective_by_group(s0, group, args.tau)
    md = [
        f"# {args.name}\n",
        f"Model: `{args.model}` (temperature {getattr(model, 'temperature', 1.0):.3g}); "
        f"windows: {len(s0)}, subjects: {s0['subject'].nunique()}, group: `{group}`\n",
        "## Subject-level HR MAE by group (95% subject-bootstrap CI)\n",
        to_markdown(t, ["method"], group),
        f"\n\n## Selective prediction at tau = {args.tau}\n",
        df_markdown(sel),
        f"\n\nCoverage parity ratio: **{coverage_parity_ratio(sel):.3f}**; "
        f"ECE(|err|<=5 bpm): {ece(s0['conf'], s0['abs_err'] <= 5):.3f}\n",
    ]
    if len(args.severities) > 1:
        sev = df.groupby("severity").apply(
            lambda d: d.groupby("subject")["abs_err"].mean().mean(), include_groups=False
        )
        md += ["\n## Robustness (IllumSim-IN)\n", df_markdown(sev.to_frame("subject_MAE"))]
        plot_severity(df, "method", out / "mae_vs_severity.png", "TraceNet robustness")
    plot_group_bars(t, group, "method", out / "mae_by_group.png", f"TraceNet by {group}")
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/ and {pred_path}")


if __name__ == "__main__":
    main()
