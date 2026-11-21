"""Export a trained TraceNet for the phone: ONNX fp32 (+ static INT8), parity and accuracy checks.

    uv run --no-sync python scripts/export_model.py --model runs/<run>/best.pt
    uv run --no-sync python scripts/export_model.py --model runs/<run>/best.pt --int8 \
        --cache data/cache/traces/ubfc --eval-cache data/cache/traces/mmpd

Writes outputs/export/<name>/{tracenet.onnx, tracenet_int8.onnx, model_meta.json, golden/*.json,
export_report.md}. Golden files (synthetic raw buffers + expected window + expected outputs) are the
parity fixtures for the Android port.
INT8 is calibrated on --cache windows (training data, never the test set); --eval-cache measures the
fp32 -> INT8 HR MAE change (M4 exit: within 1 bpm). Copy the .onnx + model_meta.json into
android/app/src/main/assets/ for the app.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pulsar.data.loading import load_cache_windows, synthetic_windows
from pulsar.eval.report import df_markdown
from pulsar.export.golden import check_golden, write_golden
from pulsar.export.onnx_export import (
    INPUT_NAMES,
    export_onnx,
    onnx_session,
    parity,
    quantize_int8,
    run_onnx,
    write_metadata,
)
from pulsar.train.dataset import WindowDataset
from pulsar.train.loop import load_model

FS = 30.0


def batches_from(ws, limit: int, seed: int = 0) -> list[tuple[torch.Tensor, ...]]:
    ds = WindowDataset(ws)
    idx = np.random.default_rng(seed).permutation(len(ds))[:limit]
    out = []
    for k in idx:
        b = ds[int(k)]
        out.append((b["x"][None], b["valid"][None].float(), b["n_pix"][None], b["motion"].view(1)))
    return out


def hr_eval(path: Path, ws) -> dict:
    """Window HR MAE of an ONNX model vs the GT HR at batch 1 (the phone's setting), plus median
    latency with 1 CPU thread (a rough lower bound for a phone big core; profile on-device for M5)."""
    ds = WindowDataset(ws)
    sess = onnx_session(path, threads=1)
    errs, t = [], []
    for k in range(len(ds)):
        b = ds[k]
        inp = (b["x"][None], b["valid"][None].float(), b["n_pix"][None], b["motion"].view(1))
        t0 = time.perf_counter()
        out = run_onnx(sess, inp)
        t.append(time.perf_counter() - t0)
        errs.append(abs(float(out[1][0]) - float(b["hr"])))
    ms = 1e3 * float(np.median(t[5:] or t))  # skip warm-up runs
    return {"mae": float(np.mean(errs)), "n": len(errs), "ms_1thread_desktop": ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--cache", nargs="*", type=Path, default=[], help="calibration windows (training data)")
    ap.add_argument("--eval-cache", nargs="*", type=Path, default=[], help="held-out windows for MAE check")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic windows for calib/eval")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = load_model(args.model, "cpu")
    name = args.name or Path(args.model).parent.name
    out = Path("outputs/export") / name
    fp32 = out / "tracenet.onnx"
    info = export_onnx(model, fp32, opset=args.opset)
    par = parity(model, fp32)
    write_metadata(model, out / "model_meta.json", {"onnx": fp32.name, "exporter": info["exporter"]})
    md = [
        f"# Export {name}\n",
        f"Source: `{args.model}`; exporter: {info['exporter']} (opset {info['opset']}); "
        f"params: {model.num_params():,}; fp32 size {fp32.stat().st_size / 1e6:.2f} MB\n",
        "## fp32 parity (max |torch - onnxruntime|)\n",
        "\n".join(f"- `{k}`: {v:.2e}" for k, v in par.items()),
    ]

    calib_ws = (
        synthetic_windows(n_subjects=6, seed=7) if args.synthetic else load_cache_windows(args.cache, FS)
    )
    eval_ws = (
        synthetic_windows(n_subjects=6, seed=8) if args.synthetic else load_cache_windows(args.eval_cache, FS)
    )
    rows = {}
    if args.int8:
        if not calib_ws:
            raise SystemExit("--int8 needs --cache (training windows) or --synthetic for calibration")
        q = quantize_int8(fp32, out / "tracenet_int8.onnx", batches_from(calib_ws, args.n_calib))
        md.append(f"\n\nINT8 (static QDQ, per-channel Conv weights): {q.stat().st_size / 1e6:.2f} MB")
    if eval_ws:
        rows["fp32"] = hr_eval(fp32, eval_ws)
        if args.int8:
            rows["int8"] = hr_eval(out / "tracenet_int8.onnx", eval_ws)
        import pandas as pd

        t = pd.DataFrame(rows).T
        md += ["\n\n## Window HR MAE (batch 1)\n", df_markdown(t)]
        if "int8" in rows:
            d = rows["int8"]["mae"] - rows["fp32"]["mae"]
            md.append(
                f"\n\nINT8 - fp32 MAE: **{d:+.2f} bpm** "
                f"({'PASS' if d <= 1.0 else 'FAIL'} vs the 1 bpm budget)"
            )
    golden = write_golden(fp32, out / "golden")
    worst = {}
    for p in golden:
        for k, v in check_golden(p, fp32).items():
            worst[k] = max(worst.get(k, 0.0), v)
    md += [
        f"\n\n## Golden files ({len(golden)} synthetic cases in `golden/`)\n",
        "Self-check max error: " + ", ".join(f"`{k}` {v:.1e}" for k, v in worst.items()),
    ]
    md.append(
        f"\n\nInputs: {', '.join(INPUT_NAMES)} (see model_meta.json). Copy the .onnx and model_meta.json "
        "into android/app/src/main/assets/ and golden/*.json into android/app/src/androidTest/assets/."
    )
    (out / "export_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
