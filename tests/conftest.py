from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def canonical_vertices() -> np.ndarray:
    """MediaPipe canonical face model vertices [468, 3]: x lateral, y up, z toward camera."""
    return np.loadtxt(ASSETS / "mediapipe_canonical_face_vertices.csv", delimiter=",")


def canonical_landmarks_norm(v: np.ndarray, width: int, height: int, face_frac: float = 0.5):
    """Project the canonical face frontally into an image -> MediaPipe-style normalised landmarks.

    Image x grows right, y grows down; MediaPipe z is negative toward the camera.
    """
    xy = v[:, :2]
    span = xy[:, 0].max() - xy[:, 0].min()
    scale_px = face_frac * width / span
    cx, cy = width / 2, height / 2
    x_px = cx + v[:, 0] * scale_px
    y_px = cy - (v[:, 1] - xy[:, 1].mean()) * scale_px
    z_px = -v[:, 2] * scale_px
    return np.stack([x_px / width, y_px / height, z_px / width], axis=1)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
