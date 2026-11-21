"""Evaluate training-free baselines (POS, CHROM, GREEN, LGI, ICA) on cached traces.

    uv run python scripts/run_classical.py --cache data/cache/traces/ubfc data/cache/traces/pure
    uv run python scripts/run_classical.py --cache data/cache/traces/ubfc --mode video   # toolbox-style

Window mode (Pulsar protocol): 10 s windows, 1 s stride, exact 1-bpm HR grid, 0.7-3.5 Hz.
Video mode (toolbox parity): whole video, toolbox detrend + 0.75-2.5 Hz filter, FFT peak.
Writes per-window/per-video predictions to outputs/classical/<name>.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pulsar.data.bundle import TraceBundle
from pulsar.data.gt import gt_on_grid
from pulsar.data.windows import make_windows
from pulsar.eval.metrics import hr_metrics, subject_level
from pulsar.preprocess.resample import resample
from pulsar.signal.classical import METHODS, run_method
from pulsar.signal.filters import postprocess_bvp
from pulsar.signal.spectral import hr_fft_toolbox, hr_grid, snr_db, snr_db_toolbox

FS = 30.0
META_COLS = ("fitzpatrick", "light", "motion", "exercise", "activity", "task")


def _subject_ita(b: TraceBundle) -> float:
    ita = b.ita_patch[np.isfinite(b.ita_patch)]
    return float(np.median(ita)) if ita.size else float("nan")


def _base_row(b: TraceBundle) -> dict:
    row = {
        "dataset": b.meta.get("dataset"),
        "video_id": b.meta.get("video_id"),
        "subject": f"{b.meta.get('dataset')}:{b.meta.get('subject')}",
        "ita": _subject_ita(b),
    }
    for k in META_COLS:
        if k in b.meta:
            v = b.meta[k]
            row[k] = v if isinstance(v, (int, float, str)) else str(v)
    return row


def eval_windows(b: TraceBundle, methods: list[str], win_s: float, stride_s: float) -> list[dict]:
    w = make_windows(b, FS, win_s, stride_s)
    rgb = w.face_mean_rgb()
    base = _base_row(b)
    rows = []
    for i in range(len(w)):
        r = rgb[i]
        if not np.isfinite(r).all():
            continue
        for m in methods:
            bvp = postprocess_bvp(run_method(m, r, FS), FS, mode="pulsar")
            rows.append(
                base
                | {
                    "method": m,
                    "start_s": float(w.start_s[i]),
                    "hr_true": float(w.hr[i]),
                    "hr_pred": hr_grid(bvp, FS),
                    "snr": snr_db(bvp, float(w.hr[i]), FS),
                    "gt_snr": float(w.gt_snr[i]),
                    "motion_px": float(w.motion[i]),
                }
            )
    return rows


def eval_video(b: TraceBundle, methods: list[str]) -> list[dict]:
    rgb = np.where(b.valid[..., None], b.rgb_lin, np.nan)
    with np.errstate(invalid="ignore"):
        face = np.nanmean(rgb, axis=1)
    ok = np.isfinite(face).all(axis=1) & b.face_found
    tg, x, _ = resample(b.t[ok], face[ok], fs=FS)
    gt = postprocess_bvp(gt_on_grid(b, tg), FS, mode="toolbox")
    hr_true = hr_fft_toolbox(gt, FS)
    rows = []
    for m in methods:
        bvp = postprocess_bvp(run_method(m, x, FS), FS, mode="toolbox")
        rows.append(
            _base_row(b)
            | {
                "method": m,
                "hr_true": hr_true,
                "hr_pred": hr_fft_toolbox(bvp, FS),
                "snr": snr_db_toolbox(bvp, hr_true, FS),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", type=Path, required=True)
    ap.add_argument(
        "--methods", nargs="+", default=["POS", "CHROM", "GREEN", "LGI", "ICA"], choices=sorted(METHODS)
    )
    ap.add_argument("--mode", choices=["window", "video"], default="window")
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--group", default=None, help="extra grouping column, e.g. fitzpatrick")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(f for c in args.cache for f in c.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {args.cache}")
    rows: list[dict] = []
    for f in files:
        b = TraceBundle.load(f)
        rows += (
            eval_video(b, args.methods)
            if args.mode == "video"
            else eval_windows(b, args.methods, args.win, args.stride)
        )
    df = pd.DataFrame(rows)
    name = "_".join(c.name for c in args.cache) + f"_{args.mode}"
    out = args.out or Path("outputs/classical") / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print(f"\n{len(files)} videos, {df['subject'].nunique()} subjects -> {out}\n")
    summary = []
    for (ds, m), d in df.groupby(["dataset", "method"]):
        s = subject_level(d)
        pooled = hr_metrics(d["hr_pred"], d["hr_true"])
        summary.append(
            {
                "dataset": ds,
                "method": m,
                "subject_MAE": s["mae"].iloc[0],
                "pooled_MAE": pooled["mae"],
                "RMSE": pooled["rmse"],
                "r": pooled["pearson"],
                "SNR_dB": d["snr"].median(),
                "n": pooled["n"],
            }
        )
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    if args.group and args.group in df:
        print(f"\nSubject-level MAE by {args.group}:")
        for m, d in df.groupby("method"):
            print(f"-- {m}\n{subject_level(d, args.group).round(2).to_string()}")


if __name__ == "__main__":
    main()
