"""Model-independent port fixtures: raw inputs + Python-reference outputs for the Android core.

Both kinds are synthetic (no dataset content), so they are committed as test resources of
``android/core`` (``scripts/make_android_fixtures.py`` regenerates them):

* ``window_<case>.json.gz``: raw ring buffer -> live model window (resampler / live_window parity).
* ``extractor.json.gz``: synthetic face images + MediaPipe-style landmarks -> per-frame patch stats
  (LUT, boxes, view angles, skin model, masks, carry-over, motion).

The model golden files (``pulsar.export.golden``) add ONNX outputs on top of ``window_case``.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
from pathlib import Path

import numpy as np

from pulsar.data.synthetic import SKIN_PRESETS, synthetic_bundle
from pulsar.preprocess.color import srgb8_to_linear_lut
from pulsar.preprocess.live import live_window, window_contract
from pulsar.preprocess.patches import PatchSpec, landmarks_to_pixels, patch_boxes
from pulsar.preprocess.traces import TraceExtractor, extractor_contract

FIXTURE_VERSION = 1
WINDOW_TOLERANCE = {"x": 1e-5, "n_pix": 1e-3, "motion": 1e-6}
WINDOW_CASES = (  # name, skin, hr, jitter_ms, drop_frac, invalid patches
    ("light_clean", "light", 68.0, 1.0, 0.0, ()),
    ("dark_jitter_drops", "dark", 104.0, 6.0, 0.05, (0, 1, 2)),
    ("medium_gaps", "medium", 82.0, 3.0, 0.0, (5,)),
)
EXTRACTOR_TOLERANCE = {
    "rgb": 1e-5,  # patch means (linear and sRGB), absolute
    "n_pix": 1,  # skin pixels per patch (a pixel exactly on a mask threshold may flip)
    "view_angle_deg": 1e-4,
    "motion": 1e-7,
    "skin_mean": 1e-6,
    "skin_cov": 1e-9,
    "lum_median": 1e-6,
}
CANONICAL_CSV = Path(__file__).resolve().parents[3] / "tests/assets/mediapipe_canonical_face_vertices.csv"


def _r(a, nd: int = 9) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _nan_to_none(a) -> list:
    """Nested list with NaN -> None (strict JSON)."""
    out = np.asarray(a, dtype=np.float64).tolist()

    def fix(v):
        if isinstance(v, list):
            return [fix(u) for u in v]
        return None if (v is None or math.isnan(v)) else v

    return fix(out)


# ------------------------------------------------------------------------------------------ windows


def window_case(
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
) -> tuple[dict, tuple]:
    """Synthetic ring buffer -> (JSON-ready case, live window computed from the *serialised* raw)."""
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
    win = live_window(rt["t"], rt["rgb"], rt["valid"], rt["n_pix"], rt["motion"], fs, win_s)
    tg, x, v, npx, mot = win
    case = {
        "name": name,
        "description": f"synthetic {skin} skin, {hr_bpm:.0f} bpm, jitter {jitter_ms} ms, "
        f"drops {drop_frac:.0%}, invalid patches {list(invalid)}",
        "fs_hz": fs,
        "win_s": win_s,
        "window": window_contract(fs, win_s),
        "raw": raw_json,
        "expected": {
            "t_grid": _r(tg),
            "x": _r(x),
            "valid": v.astype(int).tolist(),
            "n_pix": _r(npx),
            "motion": float(mot),
        },
        "tolerance": dict(WINDOW_TOLERANCE),
    }
    return case, win


# ---------------------------------------------------------------------------------------- extractor


def canonical_vertices(path: Path = CANONICAL_CSV) -> np.ndarray:
    """MediaPipe canonical face model vertices [468, 3] (x lateral, y up, z toward camera)."""
    return np.loadtxt(path, delimiter=",")


def project_face(
    v: np.ndarray,
    width: int,
    height: int,
    face_frac: float = 0.85,
    yaw_deg: float = 0.0,
    shift_px: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Canonical vertices -> MediaPipe-style normalised landmarks (image y down, z < 0 toward camera).

    The scale is fixed by the *unrotated* face width, so a yaw only turns the face.
    """
    span = v[:, 0].max() - v[:, 0].min()
    s = face_frac * width / span
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    c = v - v.mean(axis=0)
    vr = c @ rot.T
    x_px = width / 2 + vr[:, 0] * s + shift_px[0]
    y_px = height / 2 - vr[:, 1] * s + shift_px[1]
    z_px = -vr[:, 2] * s
    return np.stack([x_px / width, y_px / height, z_px / width], axis=1)


def synthetic_face_image(
    lm_norm: np.ndarray,
    width: int,
    height: int,
    skin_srgb: tuple[float, float, float],
    rng: np.random.Generator,
    spec: PatchSpec,
) -> np.ndarray:
    """uint8 [H, W, 3]: noisy skin box + hair band + nose glints + a non-skin blob on one cheek."""
    img = 30.0 + rng.integers(-2, 3, (height, width, 3)).astype(np.float64)
    px = landmarks_to_pixels(lm_norm, width, height)
    x0, y0 = np.floor(px[:, :2].min(0)).astype(int)
    x1, y1 = np.ceil(px[:, :2].max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    face = np.asarray(skin_srgb, np.float64) + rng.normal(0.0, [3.0, 2.5, 2.5], (y1 - y0, x1 - x0, 3))
    img[y0:y1, x0:x1] = face
    img[y0 : y0 + int(0.12 * (y1 - y0)), x0:x1] = (25.0, 18.0, 14.0)  # hair: dark, relative-luminance reject
    boxes = patch_boxes(px, spec, width, height)
    ids = list(spec.landmark_ids)
    for lid in (6, 197, 195, 5):  # nose glints: bright + desaturated -> specular reject
        bx0, by0, bx1, by1 = boxes[ids.index(lid)]
        g = rng.random((by1 - by0, bx1 - bx0)) < 0.06
        img[by0:by1, bx0:bx1][g] = (250.0, 250.0, 248.0)
    bx0, by0, bx1, by1 = boxes[ids.index(205)]  # green-ish blob on the right cheek -> chroma reject
    img[by0 : by0 + 6, bx0 : bx0 + 6] = (110.0, 150.0, 100.0)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def extractor_case(width: int = 240, height: int = 288, seed: int = 7) -> dict:
    """A 10-frame sequence covering detection, shift, carry-over, yaw (view-angle rejects), a skin
    colour change (skin-model momentum), carry exhaustion and re-detection."""
    spec = PatchSpec()
    v = canonical_vertices()
    rng = np.random.default_rng(seed)
    lms = {
        "frontal": project_face(v, width, height),
        "shifted": project_face(v, width, height, shift_px=(1.5, -0.8)),
        "yaw35": project_face(v, width, height, yaw_deg=35.0),
    }
    images = [
        synthetic_face_image(lms["frontal"], width, height, (200, 150, 120), rng, spec),
        synthetic_face_image(lms["shifted"], width, height, (202, 151, 121), rng, spec),
        synthetic_face_image(lms["yaw35"], width, height, (200, 150, 120), rng, spec),
        synthetic_face_image(lms["frontal"], width, height, (150, 100, 75), rng, spec),
    ]
    seq = [  # (image index, landmark key or None)
        (0, "frontal"),
        (1, "shifted"),
        (1, None),  # carried
        (2, "yaw35"),
        (3, "frontal"),
        (3, None),
        (3, None),
        (3, None),  # carried x3
        (3, None),  # carry exhausted -> empty frame
        (0, "frontal"),  # re-detected
    ]
    ex = TraceExtractor(spec)
    frames = []
    for img_i, key in seq:
        st = ex.process(images[img_i], None if key is None else lms[key])
        sk = ex.skin
        frames.append(
            {
                "image": img_i,
                "landmarks": key,
                "expected": {
                    "face_found": bool(st.face_found),
                    "motion": float(st.motion),
                    "valid": st.valid.astype(int).tolist(),
                    "n_pix": st.n_pix.astype(int).tolist(),
                    "rgb_lin": _nan_to_none(st.rgb_lin),
                    "rgb_srgb": _nan_to_none(st.rgb_srgb),
                    "boxes": None if ex.last_boxes is None else ex.last_boxes.astype(int).tolist(),
                    "view_angles": None if ex.last_angles is None else ex.last_angles.tolist(),
                    "skin": None
                    if sk.mean is None
                    else {
                        "mean": np.asarray(sk.mean, np.float64).tolist(),
                        "cov": np.asarray(sk.cov, np.float64).tolist(),
                        "lum_median": float(sk.lum_median),
                    },
                },
            }
        )
    return {
        "version": FIXTURE_VERSION,
        "description": "synthetic face (MediaPipe canonical mesh, no real faces) -> TraceExtractor",
        "width": width,
        "height": height,
        "patch_spec": spec.contract(),
        "extractor": extractor_contract(),
        "srgb_lut": srgb8_to_linear_lut().astype(np.float64).tolist(),
        "images_rgb8_b64": [base64.b64encode(np.ascontiguousarray(im).tobytes()).decode() for im in images],
        "landmarks": {k: lm.tolist() for k, lm in lms.items()},
        "frames": frames,
        "tolerance": dict(EXTRACTOR_TOLERANCE),
    }


# -------------------------------------------------------------------------------------------- write


def write_json_gz(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with gzip.GzipFile(path, "wb", mtime=0) as f:  # mtime=0: byte-identical on regeneration
        f.write(data)
    return path


def write_fixtures(out_dir: str | Path) -> list[Path]:
    """Write all fixtures plus ``index.json`` (file list; resources in a jar cannot be listed)."""
    out_dir = Path(out_dir)
    paths = [write_json_gz(extractor_case(), out_dir / "extractor.json.gz")]
    for i, (name, skin, hr, jit, drop, invalid) in enumerate(WINDOW_CASES):
        case, _ = window_case(name, skin, hr, jit, drop, invalid, seed=100 + i)
        paths.append(write_json_gz(case, out_dir / f"window_{name}.json.gz"))
    index = {
        "version": FIXTURE_VERSION,
        "generator": "scripts/make_android_fixtures.py",
        "extractor": "extractor.json.gz",
        "windows": [p.name for p in paths[1:]],
    }
    idx = out_dir / "index.json"
    idx.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return [*paths, idx]
