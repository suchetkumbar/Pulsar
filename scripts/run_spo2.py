"""M4 experimental SpO2: ratio-of-ratios vs a constant-mean baseline (docs 04 §5).

    uv run --no-sync python scripts/run_spo2.py --cache data/cache/traces/pure
    uv run --no-sync python scripts/run_spo2.py --cache data/cache/traces/pure --model runs/<run>/best.pt
    uv run --no-sync python scripts/run_spo2.py --synthetic   # estimator check on a toy model -> outputs/

SpO2 = A + B*RR is calibrated leave-one-subject-out. The AC reference is POS on the face mean, or
TraceNet's predicted BVP with --model. Writes reports/<name>/README.md (+ figures) and
outputs/<name>_features.parquet. Only PURE has SpO2 labels among the planned datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pulsar.data.gt import apply_alignment, estimate_alignment  # noqa: E402
from pulsar.data.loading import bundle_to_windows, load_cache_windows  # noqa: E402
from pulsar.data.synthetic import SKIN_PRESETS, synthetic_bundle  # noqa: E402
from pulsar.data.windows import Windows  # noqa: E402
from pulsar.eval.report import df_markdown  # noqa: E402
from pulsar.vitals.spo2 import evaluate_loso, loso_predict, window_features  # noqa: E402

FS = 30.0
CAVEATS = """## Read this before quoting any number
- RGB cameras have no red/infrared pair; the R/B and R/G ratios are weak proxies for oxygenation.
- PURE's reference SpO2 spans a narrow, normoxic range, so the **constant-mean predictor is the bar
  to beat**, and correlation matters more than MAE. Hypoxaemic values (< 90 %) are untested.
- When SpO2 barely changes *within* a subject, any correlation is mostly *between* subjects and can
  be driven by skin tone, lighting or camera position rather than oxygenation (see the variance split).
- The reference oximeter is itself biased on darker skin (Sjoding et al., NEJM 2020).
- Research probe only; not a measurement and not a medical device.
"""


def synthetic_cohort(n_subjects: int = 12, seed: int = 0) -> list[Windows]:
    rng = np.random.default_rng(seed)
    light, dark = np.log(SKIN_PRESETS["light"]), np.log(SKIN_PRESETS["dark"])
    out = []
    for s in range(n_subjects):
        tone = s / max(1, n_subjects - 1)
        for v in range(2):
            b = synthetic_bundle(
                duration_s=40.0,
                hr_bpm=float(rng.uniform(55, 110)),
                skin=np.exp((1 - tone) * light + tone * dark),
                spo2=float(rng.uniform(90, 99)),
                noise_a=2e-3,
                n_pix=int(rng.integers(80, 400)),
                seed=seed * 10_000 + s * 100 + v,
            )
            b.meta.update(subject=f"s{s}", video_id=f"s{s}_v{v}")
            apply_alignment(b, estimate_alignment(b))
            out.append(bundle_to_windows(b, FS, 10.0, 2.0))
    return out


def tracenet_refs(ws: list[Windows], model_path: str, device_name: str) -> list[np.ndarray]:
    """Predicted BVP per window (same order as ``ws``), used as the matched-filter reference."""
    import torch
    from torch.utils.data import DataLoader

    from pulsar.train.dataset import WindowDataset
    from pulsar.train.loop import load_model, resolve_device

    device = resolve_device(device_name)
    model = load_model(model_path, device)
    ds = WindowDataset(ws)
    bvp = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=128, shuffle=False):
            out = model(
                b["x"].to(device), b["valid"].to(device), b["n_pix"].to(device), b["motion"].to(device)
            )
            bvp.append(out["bvp"].float().cpu().numpy())
    flat = np.concatenate(bvp)
    sizes = np.cumsum([0] + [len(w) for w in ds.windows])
    return [flat[sizes[i] : sizes[i + 1]] for i in range(len(ds.windows))]


def variance_split(df: pd.DataFrame) -> dict[str, float]:
    per = df.groupby("subject")["spo2_true"]
    return {
        "range_lo": float(df["spo2_true"].min()),
        "range_hi": float(df["spo2_true"].max()),
        "between_subject_sd": float(per.mean().std(ddof=1)),
        "within_subject_sd": float(per.std(ddof=1).mean()),
    }


def plots(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for s, d in df.groupby("subject"):
        ax[0].scatter(d["rr_rb"], d["spo2_true"], s=8, alpha=0.6, label=s)
        ax[1].scatter(d["spo2_true"], d["pred_rr_rb"], s=8, alpha=0.6)
    ax[0].set_xlabel("ratio-of-ratios R/B")
    ax[0].set_ylabel("reference SpO2 (%)")
    ax[0].set_title("Feature vs reference (colour = subject)", fontsize=9)
    lo, hi = df["spo2_true"].min() - 1, df["spo2_true"].max() + 1
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[1].axhline(df["pred_const"].mean(), color="0.5", lw=0.8, ls=":", label="constant-mean")
    ax[1].set_xlabel("reference SpO2 (%)")
    ax[1].set_ylabel("LOSO prediction (%)")
    ax[1].set_title("Leave-one-subject-out, RoR R/B", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "spo2_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*", type=Path, default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default=None, help="TraceNet checkpoint: use its BVP as the AC reference")
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    ws = synthetic_cohort() if args.synthetic else load_cache_windows(args.cache, FS, 10.0, args.stride)
    ws = [w for w in ws if len(w) and w.spo2 is not None]
    if not ws:
        raise SystemExit("no windows with SpO2 ground truth (of the planned datasets only PURE has it)")
    ref_name = "tracenet" if args.model else "pos"
    refs = tracenet_refs(ws, args.model, args.device) if args.model else None
    df = window_features(ws, FS, refs=refs, ref_name=ref_name)
    df = df[np.isfinite(df["spo2_true"])].reset_index(drop=True)
    res = evaluate_loso(df)
    df["pred_const"] = loso_predict(df, None)
    df["pred_rr_rb"] = loso_predict(df, "rr_rb")

    name = args.name or ("m4_spo2_synthetic" if args.synthetic else f"m4_spo2_{ref_name}")
    out = (Path("outputs") if args.synthetic else Path("reports")) / name
    out.mkdir(parents=True, exist_ok=True)
    feat_path = Path("outputs") / f"{name}_features.parquet"
    df.to_parquet(feat_path)
    vs = variance_split(df)
    md = [
        f"# {name}\n",
        (
            "**Synthetic toy model** (SpO2 linearly scales the red pulse weight); this checks the estimator, "
            "not physiology.\n"
            if args.synthetic
            else ""
        ),
        f"AC reference: `{ref_name}`; windows: {len(df)}; subjects: {df['subject'].nunique()}; "
        f"reference SpO2 range {vs['range_lo']:.0f}-{vs['range_hi']:.0f} %; "
        f"between-subject SD {vs['between_subject_sd']:.2f}, "
        f"mean within-subject SD {vs['within_subject_sd']:.2f}.\n",
        "## Leave-one-subject-out calibration\n",
        df_markdown(res),
        "\n\nSubject-level MAE is the primary number; Pearson is window-pooled. The constant-mean "
        "predictor's negative r is a known leave-one-out artefact (removing a high subject lowers the "
        "training mean), not a signal.\n",
        CAVEATS,
    ]
    plots(df, out)
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out}/ and {feat_path}")


if __name__ == "__main__":
    main()
