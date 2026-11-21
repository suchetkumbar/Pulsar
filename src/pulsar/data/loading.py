"""Load cached TraceBundles (or synthetic ones) into windows, with subject-wise splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pulsar.data.bundle import TraceBundle
from pulsar.data.gt import apply_alignment, estimate_alignment
from pulsar.data.synthetic import SKIN_PRESETS, synthetic_bundle
from pulsar.data.windows import Windows, make_windows


def subject_ita(b: TraceBundle) -> float:
    ita = b.ita_patch[np.isfinite(b.ita_patch)]
    return float(np.median(ita)) if ita.size else float("nan")


def bundle_to_windows(b: TraceBundle, fs: float, win_s: float, stride_s: float) -> Windows:
    w = make_windows(b, fs, win_s, stride_s)
    w.meta["ita"] = subject_ita(b)
    return w


def load_cache_windows(cache_dirs: list[str | Path], fs=30.0, win_s=10.0, stride_s=1.0) -> list[Windows]:
    out = []
    for d in cache_dirs:
        for f in sorted(Path(d).glob("*.npz")):
            out.append(bundle_to_windows(TraceBundle.load(f), fs, win_s, stride_s))
    return out


def synthetic_windows(
    n_subjects: int = 24,
    videos_per_subject: int = 2,
    duration_s: float = 40.0,
    fs: float = 30.0,
    win_s: float = 10.0,
    stride_s: float = 2.0,
    seed: int = 0,
) -> list[Windows]:
    """Synthetic cohort spanning light->dark skin and 55-120 bpm, for smoke tests/debugging."""
    rng = np.random.default_rng(seed)
    light, dark = SKIN_PRESETS["light"], SKIN_PRESETS["dark"]
    out = []
    for s in range(n_subjects):
        tone = s / max(1, n_subjects - 1)  # 0 = light .. 1 = dark (log-interpolated)
        skin = np.exp((1 - tone) * np.log(light) + tone * np.log(dark))
        fitz = int(np.clip(np.round(3 + 3 * tone), 3, 6))
        for v in range(videos_per_subject):
            b = synthetic_bundle(
                duration_s=duration_s,
                fs=fs,
                hr_bpm=float(rng.uniform(55, 120)),
                skin=skin,
                seed=seed * 10_000 + s * 100 + v,
                noise_a=2e-3,
                n_pix=int(rng.integers(80, 400)),
            )
            b.meta.update(subject=f"s{s}", video_id=f"s{s}_v{v}", fitzpatrick=fitz)
            apply_alignment(b, estimate_alignment(b))
            out.append(bundle_to_windows(b, fs, win_s, stride_s))
    return out


def split_subjects(
    ws: list[Windows],
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
    split_file: str | Path | None = None,
) -> tuple[list[Windows], list[Windows], list[Windows]]:
    """Subject-wise split. If split_file exists ({"train": [...], "val": [...], "test": [...]} of
    "dataset:subject" ids) it is used verbatim; otherwise a seeded random split is drawn."""
    key = lambda w: f"{w.meta.get('dataset')}:{w.subject}"  # noqa: E731
    if split_file and Path(split_file).exists():
        sp = json.loads(Path(split_file).read_text())
        parts = {k: set(v) for k, v in sp.items()}
        return tuple([w for w in ws if key(w) in parts.get(name, set())] for name in ("train", "val", "test"))
    subjects = sorted({key(w) for w in ws})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_te = int(round(test_frac * len(subjects)))
    n_va = max(1, int(round(val_frac * len(subjects)))) if val_frac > 0 else 0
    te, va = set(subjects[:n_te]), set(subjects[n_te : n_te + n_va])
    return (
        [w for w in ws if key(w) not in te | va],
        [w for w in ws if key(w) in va],
        [w for w in ws if key(w) in te],
    )
