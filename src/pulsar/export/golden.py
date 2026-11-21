"""Golden files: raw ring-buffer inputs + expected resampled window + expected ONNX outputs, as JSON.

An Android instrumented test (or any port) loads a case, runs its own resampler and ONNX Runtime on
``raw`` and must reproduce ``expected`` within ``tolerance`` (docs 07 §4). Inputs are synthetic only
(no dataset-derived data), so golden files are safe to commit and ship as test assets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pulsar.data.synthetic import SKIN_PRESETS, synthetic_bundle
from pulsar.export.onnx_export import OUTPUT_NAMES, run_onnx
from pulsar.preprocess.live import live_window

TOLERANCE = {"x": 1e-5, "n_pix": 1e-3, "motion": 1e-6, "bvp": 2e-3, "hr": 0.05, "conf": 2e-3, "alpha": 1e-4}
CASES = (  # name, skin, hr, jitter_ms, drop_frac, invalid patches
    ("light_clean", "light", 68.0, 1.0, 0.0, ()),
    ("dark_jitter_drops", "dark", 104.0, 6.0, 0.05, (0, 1, 2)),
    ("medium_gaps", "medium", 82.0, 3.0, 0.0, (5,)),
)


def _r(a, nd: int = 9) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def model_inputs(x, v, npx, mot) -> tuple[torch.Tensor, ...]:
    """Live window -> exported-graph float32 inputs: x [1,P,T,3], valid [1,P,T], n_pix [1,P], motion [1]."""
    return (
        torch.from_numpy(np.ascontiguousarray(x.transpose(1, 0, 2)[None])).float(),
        torch.from_numpy(np.ascontiguousarray(v.T[None])).float(),
        torch.from_numpy(npx[None]).float(),
        torch.tensor([mot], dtype=torch.float32),
    )


def golden_case(
    onnx_path: str | Path,
    name: str,
    skin: str,
    hr_bpm: float,
    jitter_ms: float,
    drop_frac: float,
    invalid: tuple[int, ...],
    seed: int = 0,
    buffer_s: float = 12.0,
    fs: float = 30.0,
    win_s: float = 10.0,
) -> dict:
    b = synthetic_bundle(
        duration_s=buffer_s,
        fs=fs,
        hr_bpm=hr_bpm,
        skin=SKIN_PRESETS[skin],
        jitter_ms=jitter_ms,
        drop_frac=drop_frac,
        noise_a=2e-3,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    valid = b.valid.copy()
    valid[:, list(invalid)] = False
    if name.endswith("gaps"):  # a 0.4 s face loss on some patches exercises the gap rule (> 0.25 s)
        g = (b.t > 4.0) & (b.t < 4.4)
        valid[np.ix_(g, np.arange(8, 16))] = False
    n_pix = b.n_pix.astype(np.float64) + rng.integers(-20, 20, b.n_pix.shape)
    motion = np.abs(rng.normal(0.0, 0.003, b.t.size))
    raw = {"t": b.t, "rgb": b.rgb_lin.astype(np.float64), "valid": valid, "n_pix": n_pix, "motion": motion}
    # round-trip the raw inputs through their JSON precision first, so expected == f(serialised raw)
    raw_json = {k: (np.asarray(v).astype(int).tolist() if k == "valid" else _r(v)) for k, v in raw.items()}
    rt = {k: np.asarray(v, dtype=bool if k == "valid" else np.float64) for k, v in raw_json.items()}
    tg, x, v, npx, mot = live_window(rt["t"], rt["rgb"], rt["valid"], rt["n_pix"], rt["motion"], fs, win_s)
    outs = dict(zip(OUTPUT_NAMES, run_onnx(onnx_path, model_inputs(x, v, npx, mot)), strict=True))
    return {
        "name": name,
        "description": f"synthetic {skin} skin, {hr_bpm:.0f} bpm, jitter {jitter_ms} ms, "
        f"drops {drop_frac:.0%}, "
        f"invalid patches {list(invalid)}",
        "fs_hz": fs,
        "win_s": win_s,
        "raw": raw_json,
        "expected": {
            "t_grid": _r(tg),
            "x": _r(x),
            "valid": v.astype(int).tolist(),
            "n_pix": _r(npx),
            "motion": float(mot),
            "outputs": {k: _r(o.ravel()) for k, o in outs.items() if k != "hr_logits"},
        },
        "tolerance": TOLERANCE,
    }


def write_golden(onnx_path: str | Path, out_dir: str | Path, cases=CASES) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, skin, hr, jit, drop, invalid) in enumerate(cases):
        case = golden_case(onnx_path, name, skin, hr, jit, drop, invalid, seed=100 + i)
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(case, separators=(",", ":")), encoding="utf-8")
        paths.append(p)
    return paths


def check_golden(path: str | Path, onnx_path: str | Path) -> dict[str, float]:
    """Re-run the Python reference on a golden file's raw inputs; returns max error per checked field."""
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    r = case["raw"]
    win = live_window(
        np.asarray(r["t"]),
        np.asarray(r["rgb"]),
        np.asarray(r["valid"], bool),
        np.asarray(r["n_pix"]),
        np.asarray(r["motion"]),
        case["fs_hz"],
        case["win_s"],
    )
    tg, x, v, npx, mot = win
    e = case["expected"]
    err = {
        "x": float(np.abs(x - np.asarray(e["x"])).max()),
        "valid": float(np.abs(v.astype(int) - np.asarray(e["valid"])).max()),
        "n_pix": float(np.abs(npx - np.asarray(e["n_pix"])).max()),
        "motion": abs(mot - e["motion"]),
    }
    outs = dict(zip(OUTPUT_NAMES, run_onnx(onnx_path, model_inputs(x, v, npx, mot)), strict=True))
    for k, want in e["outputs"].items():
        err[k] = float(np.abs(outs[k].ravel() - np.asarray(want)).max())
    return err
