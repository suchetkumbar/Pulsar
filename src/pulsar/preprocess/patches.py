"""Landmark-anchored skin patches on the MediaPipe Face Mesh (478 landmarks).

Each patch is a square centred on one mesh landmark, with side = `size_frac` x face width
(distance between the face-oval landmarks 234 and 454). Landmark indices were checked against
MediaPipe's canonical face model (tests/test_patches.py); they sit on forehead, glabella, nose
dorsum, cheeks and chin, away from eyes, brows, lips and nostrils.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 234, 454  # face-oval extremes (subject's right / left)

# region -> landmark indices (subject-centric: "_r" = subject's right = image left, unmirrored)
DEFAULT_REGIONS: dict[str, list[int]] = {
    "forehead": [151, 108, 337, 109, 338, 69, 299, 104, 333],
    "glabella": [9, 8],
    "nose": [6, 197, 195, 5],
    "cheek_r": [50, 101, 205, 117, 123, 187, 36],
    "cheek_l": [280, 330, 425, 346, 352, 411, 266],
    "chin": [200, 199, 175],
}
REGION_NAMES: list[str] = list(DEFAULT_REGIONS)


@dataclass(frozen=True)
class PatchSpec:
    regions: dict[str, list[int]] = field(default_factory=lambda: dict(DEFAULT_REGIONS))
    size_frac: float = 0.09  # patch side / face width
    min_pixels: int = 16  # minimum skin pixels for a valid patch
    max_view_angle_deg: float = 70.0  # patch normal vs camera axis
    version: str = "v1"

    @property
    def landmark_ids(self) -> np.ndarray:
        return np.array([i for ids in self.regions.values() for i in ids], dtype=np.int64)

    @property
    def region_ids(self) -> np.ndarray:
        """Integer region label per patch (index into REGION_NAMES order of self.regions)."""
        out = []
        for r, (_, ids) in enumerate(self.regions.items()):
            out += [r] * len(ids)
        return np.array(out, dtype=np.int64)

    @property
    def num_patches(self) -> int:
        return int(self.landmark_ids.shape[0])

    def to_meta(self) -> dict:
        return {
            "regions": self.regions,
            "size_frac": self.size_frac,
            "min_pixels": self.min_pixels,
            "max_view_angle_deg": self.max_view_angle_deg,
            "version": self.version,
        }


def landmarks_to_pixels(lm_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe normalised landmarks [N, 3] -> pixel-scale 3-D points [N, 3] (z scaled by width)."""
    lm = np.asarray(lm_norm, dtype=np.float64)
    return np.stack([lm[:, 0] * width, lm[:, 1] * height, lm[:, 2] * width], axis=1)


def face_width_px(lm_px: np.ndarray) -> float:
    return float(np.linalg.norm(lm_px[FACE_LEFT_EDGE, :2] - lm_px[FACE_RIGHT_EDGE, :2]))


def patch_boxes(lm_px: np.ndarray, spec: PatchSpec, width: int, height: int) -> np.ndarray:
    """Integer boxes [P, 4] = (x0, y0, x1, y1), clipped to the image (x1/y1 exclusive)."""
    centers = lm_px[spec.landmark_ids, :2]
    half = 0.5 * spec.size_frac * face_width_px(lm_px)
    half = max(half, 1.0)
    x0 = np.clip(np.floor(centers[:, 0] - half), 0, width)
    y0 = np.clip(np.floor(centers[:, 1] - half), 0, height)
    x1 = np.clip(np.ceil(centers[:, 0] + half), 0, width)
    y1 = np.clip(np.ceil(centers[:, 1] + half), 0, height)
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.int64)


def patch_view_angles(lm_px: np.ndarray, spec: PatchSpec, k: int = 10) -> np.ndarray:
    """Angle (deg) between each patch's local surface normal and the camera axis.

    Normal = smallest principal axis of the k nearest landmarks (3-D), oriented toward the camera
    (MediaPipe z decreases toward the camera, so a camera-facing normal has negative z).
    """
    ids = spec.landmark_ids
    pts = lm_px[:, :3]
    d2 = ((pts[ids, None, :] - pts[None, :, :]) ** 2).sum(-1)  # [P, N]
    nn = np.argpartition(d2, kth=min(k, pts.shape[0] - 1), axis=1)[:, : k + 1]
    angles = np.empty(len(ids))
    for p in range(len(ids)):
        q = pts[nn[p]] - pts[nn[p]].mean(axis=0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        n = vt[-1]
        if n[2] > 0:
            n = -n
        angles[p] = np.degrees(np.arccos(np.clip(-n[2] / (np.linalg.norm(n) + 1e-12), -1, 1)))
    return angles
