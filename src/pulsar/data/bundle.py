"""TraceBundle: the canonical cached representation of one video (docs 03 §4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

EXTRACT_VERSION = "2"  # 2: deterministic skin-seed subsample (Android parity)


@dataclass
class TraceBundle:
    t: np.ndarray  # [T] frame timestamps (s)
    rgb_lin: np.ndarray  # [T, P, 3] float32 patch means, linear RGB [0,1]
    rgb_srgb: np.ndarray  # [T, P, 3] float32 patch means of sRGB values [0,1]
    valid: np.ndarray  # [T, P] bool
    n_pix: np.ndarray  # [T, P] int32 skin pixels averaged
    center: np.ndarray  # [T, P, 2] float32 normalised (x, y) image coords
    motion: np.ndarray  # [T] float32 landmark motion (face-width units / frame)
    face_found: np.ndarray  # [T] bool
    ita_patch: np.ndarray  # [P] float32 median ITA per patch (deg)
    region_ids: np.ndarray  # [P] int region label per patch
    gt_ppg: np.ndarray  # [T_gt] ground-truth PPG
    gt_t: np.ndarray  # [T_gt] timestamps (s), same clock as t
    gt_spo2: np.ndarray | None = None  # [T_gt] or None
    gt_hr: np.ndarray | None = None  # [T_gt] device-reported HR (sanity only) or None
    lag_s: float = 0.0  # estimated face-vs-finger lag; GT(t + lag) aligns with face(t)
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def num_patches(self) -> int:
        return int(self.rgb_lin.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.num_frames > 1 else 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "meta":
                arrays["meta_json"] = np.array(json.dumps(v, default=_json_default))
            elif f.name == "lag_s":
                arrays["lag_s"] = np.array(v, dtype=np.float64)
            elif v is not None:
                arrays[f.name] = np.asarray(v)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TraceBundle:
        with np.load(Path(path), allow_pickle=False) as z:
            kw = {}
            for f in fields(cls):
                if f.name == "meta":
                    kw["meta"] = json.loads(str(z["meta_json"]))
                elif f.name == "lag_s":
                    kw["lag_s"] = float(z["lag_s"])
                elif f.name in z.files:
                    kw[f.name] = z[f.name]
                else:
                    kw[f.name] = None
        return cls(**kw)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
