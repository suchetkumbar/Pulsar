"""Per-frame patch averaging: video frames + landmarks -> TraceBundle (without ground truth).

Pixels are linearised (inverse sRGB LUT) *before* averaging, because the physics-based
augmentations (PhysTone-Aug, IllumSim-IN) are defined on linear intensities.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from pulsar.data.bundle import EXTRACT_VERSION, TraceBundle
from pulsar.preprocess.color import ita_from_linear_rgb, srgb8_to_linear_lut
from pulsar.preprocess.face import Landmarker
from pulsar.preprocess.patches import (
    PatchSpec,
    face_width_px,
    landmarks_to_pixels,
    patch_boxes,
    patch_view_angles,
)
from pulsar.preprocess.skin_mask import (
    CHI2_2DOF_99,
    MIN_SEED_PIXELS,
    SHADOW_FRAC,
    SKIN_MOMENTUM,
    SPECULAR_PCT,
    SPECULAR_SAT_MAX,
    SkinModel,
    specular_mask,
)

SEED_REGIONS = ("cheek_r", "cheek_l")
SEED_MAX_ANGLE_DEG = 50.0  # only near-frontal cheek patches seed the skin model
SEED_MAX_PIXELS = 4000  # evenly spaced subsample above this (deterministic: the Android port mirrors it)
CARRY_FRAMES = 3  # reuse the last landmarks for this many face-less frames


def extractor_contract() -> dict:
    """Constants a port must mirror (model_meta.json "extractor"); see android/core TraceExtractor.kt."""
    return {
        "linearize": "inverse sRGB LUT on uint8 before averaging",
        "seed_regions": list(SEED_REGIONS),
        "seed_max_angle_deg": SEED_MAX_ANGLE_DEG,
        "seed_max_pixels": SEED_MAX_PIXELS,
        "carry_frames": CARRY_FRAMES,
        "skin_momentum": SKIN_MOMENTUM,
        "skin_radius2": CHI2_2DOF_99,
        "skin_min_seed_pixels": MIN_SEED_PIXELS,
        "shadow_frac": SHADOW_FRAC,
        "specular_pct": SPECULAR_PCT,
        "specular_sat_max": SPECULAR_SAT_MAX,
    }


@dataclass
class FrameStats:
    rgb_lin: np.ndarray  # [P, 3]
    rgb_srgb: np.ndarray  # [P, 3]
    n_pix: np.ndarray  # [P]
    center: np.ndarray  # [P, 2]
    valid: np.ndarray  # [P]
    face_found: bool
    motion: float


class TraceExtractor:
    def __init__(self, spec: PatchSpec | None = None, carry_frames: int = CARRY_FRAMES):
        self.spec = spec or PatchSpec()
        self.lut = srgb8_to_linear_lut()
        self.skin = SkinModel()
        self.carry_frames = carry_frames
        self._last_lm: np.ndarray | None = None
        self._missing = 0
        self.last_boxes: np.ndarray | None = None  # [P, 4] boxes / [P] view angles of the last frame
        self.last_angles: np.ndarray | None = None
        ids = self.spec.landmark_ids
        names = list(self.spec.regions)
        rid = self.spec.region_ids
        self._seed_idx = [p for p in range(len(ids)) if names[rid[p]] in SEED_REGIONS]

    def _empty(self, motion: float = 0.0) -> FrameStats:
        p = self.spec.num_patches
        nan3 = np.full((p, 3), np.nan, dtype=np.float32)
        return FrameStats(
            nan3,
            nan3.copy(),
            np.zeros(p, np.int32),
            np.full((p, 2), np.nan, np.float32),
            np.zeros(p, bool),
            False,
            motion,
        )

    def process(self, frame_rgb: np.ndarray, lm_norm: np.ndarray | None) -> FrameStats:
        """frame_rgb: uint8 [H, W, 3] (sRGB). lm_norm: [N>=468, 3] normalised landmarks or None."""
        if frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3:
            raise ValueError("frame must be uint8 [H, W, 3]")
        found = lm_norm is not None
        if lm_norm is None:
            self._missing += 1
            if self._last_lm is None or self._missing > self.carry_frames:
                self.last_boxes = self.last_angles = None
                return self._empty()
            lm_norm = self._last_lm
        else:
            self._missing = 0
        h, w = frame_rgb.shape[:2]
        lm_px = landmarks_to_pixels(lm_norm, w, h)
        fw = max(face_width_px(lm_px), 1.0)
        motion = 0.0
        if self._last_lm is not None and found:
            prev = landmarks_to_pixels(self._last_lm, w, h)
            motion = float(np.linalg.norm(lm_px[:, :2] - prev[:, :2], axis=1).mean() / fw)
        if found:
            self._last_lm = np.asarray(lm_norm)

        spec = self.spec
        boxes = patch_boxes(lm_px, spec, w, h)
        angles = patch_view_angles(lm_px, spec)
        self.last_boxes, self.last_angles = boxes, angles
        crops_lin: list[np.ndarray] = []
        crops_u8: list[np.ndarray] = []
        for x0, y0, x1, y1 in boxes:
            c = frame_rgb[y0:y1, x0:x1].reshape(-1, 3)
            crops_u8.append(c)
            crops_lin.append(self.lut[c])

        seed = [crops_lin[p] for p in self._seed_idx if crops_lin[p].size and angles[p] < SEED_MAX_ANGLE_DEG]
        if seed:
            s = np.concatenate(seed)
            if s.shape[0] > SEED_MAX_PIXELS:
                s = s[(np.arange(SEED_MAX_PIXELS) * s.shape[0]) // SEED_MAX_PIXELS]
            self.skin.update(s)

        p_n = spec.num_patches
        rgb_lin = np.full((p_n, 3), np.nan, np.float32)
        rgb_srgb = np.full((p_n, 3), np.nan, np.float32)
        n_pix = np.zeros(p_n, np.int32)
        valid = np.zeros(p_n, bool)
        for p in range(p_n):
            lin = crops_lin[p]
            if lin.shape[0] == 0:
                continue
            m = self.skin.mask(lin) & specular_mask(lin)
            n = int(m.sum())
            n_pix[p] = n
            if n == 0:
                continue
            rgb_lin[p] = lin[m].mean(axis=0)
            rgb_srgb[p] = crops_u8[p][m].mean(axis=0) / 255.0
            valid[p] = n >= spec.min_pixels and angles[p] <= spec.max_view_angle_deg
        center = np.asarray(lm_norm)[spec.landmark_ids, :2].astype(np.float32)
        return FrameStats(rgb_lin, rgb_srgb, n_pix, center, valid, found, motion)


def extract_video(
    frames: Iterable[tuple[float, np.ndarray]],
    landmarker: Landmarker,
    spec: PatchSpec | None = None,
    meta: dict | None = None,
    progress: bool = False,
) -> TraceBundle:
    """Run landmarks + patch averaging over (timestamp_s, frame_rgb_uint8) pairs."""
    ex = TraceExtractor(spec)
    ts, stats = [], []
    it: Iterator = iter(frames)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="frames", unit="f", leave=False)
    for t_s, frame in it:
        lm = landmarker.detect(frame, int(round(t_s * 1000)))
        stats.append(ex.process(frame, lm))
        ts.append(t_s)
    if not stats:
        raise ValueError("no frames")
    rgb_lin = np.stack([s.rgb_lin for s in stats])
    valid = np.stack([s.valid for s in stats])
    ita = np.full(ex.spec.num_patches, np.nan, np.float32)
    for p in range(ex.spec.num_patches):
        v = valid[:, p]
        if v.any():
            ita[p] = float(np.median(ita_from_linear_rgb(rgb_lin[v, p])))
    m = dict(meta or {})
    m.setdefault("extract_version", EXTRACT_VERSION)
    m["patch_spec"] = ex.spec.to_meta()
    return TraceBundle(
        t=np.asarray(ts, np.float64),
        rgb_lin=rgb_lin,
        rgb_srgb=np.stack([s.rgb_srgb for s in stats]),
        valid=valid,
        n_pix=np.stack([s.n_pix for s in stats]),
        center=np.stack([s.center for s in stats]),
        motion=np.asarray([s.motion for s in stats], np.float32),
        face_found=np.asarray([s.face_found for s in stats], bool),
        ita_patch=ita,
        region_ids=ex.spec.region_ids,
        gt_ppg=np.zeros(0, np.float32),
        gt_t=np.zeros(0, np.float64),
        meta=m,
    )
