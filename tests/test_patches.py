"""Patch landmark placement, verified on MediaPipe's canonical face (x lateral, y up, z out).

Reference landmarks (canonical mesh): eye corners 33/133 & 362/263, brow tops 105/334,
mouth corners 61/291, nose tip 4, face-oval extremes 234/454.
"""

import numpy as np
import pytest

from pulsar.preprocess.patches import DEFAULT_REGIONS, PatchSpec, patch_view_angles

from .conftest import canonical_landmarks_norm

MIRROR_PAIRS = [
    (50, 280),
    (101, 330),
    (205, 425),
    (117, 346),
    (123, 352),
    (187, 411),
    (36, 266),
    (108, 337),
    (109, 338),
    (69, 299),
    (104, 333),
]


def _ref(v):
    eye_y = v[[33, 133, 362, 263], 1].mean()
    brow_y = v[[105, 334], 1].mean()
    mouth_y = v[[61, 291], 1].mean()
    return eye_y, brow_y, mouth_y


def test_mirror_pairs_are_symmetric(canonical_vertices):
    v = canonical_vertices
    for a, b in MIRROR_PAIRS:
        assert v[a, 0] == pytest.approx(-v[b, 0], abs=0.05), (a, b)
        assert v[a, 1] == pytest.approx(v[b, 1], abs=0.05), (a, b)


def test_forehead_above_brows(canonical_vertices):
    v = canonical_vertices
    _, brow_y, _ = _ref(v)
    for i in DEFAULT_REGIONS["forehead"]:
        assert v[i, 1] > brow_y, i


def test_glabella_and_nose_on_midline(canonical_vertices):
    v = canonical_vertices
    eye_y, _, mouth_y = _ref(v)
    for i in DEFAULT_REGIONS["glabella"] + DEFAULT_REGIONS["nose"]:
        assert abs(v[i, 0]) < 0.6, i
    for i in DEFAULT_REGIONS["nose"]:
        assert mouth_y < v[i, 1] < v[DEFAULT_REGIONS["glabella"][0], 1], i


def test_cheeks_lateral_below_eyes_above_mouth(canonical_vertices):
    v = canonical_vertices
    eye_y, _, mouth_y = _ref(v)
    eye_inner_x = abs(v[133, 0])
    for side in ("cheek_r", "cheek_l"):
        for i in DEFAULT_REGIONS[side]:
            assert np.sign(v[i, 0]) == np.sign(v[DEFAULT_REGIONS[side][0], 0]), i
            assert abs(v[i, 0]) > 0.6 * eye_inner_x, i
            assert mouth_y - 0.5 < v[i, 1] < eye_y - 0.5, i
    # the two cheeks are on opposite sides
    assert np.sign(v[50, 0]) != np.sign(v[280, 0])


def test_chin_below_mouth(canonical_vertices):
    v = canonical_vertices
    _, _, mouth_y = _ref(v)
    for i in DEFAULT_REGIONS["chin"]:
        assert v[i, 1] < mouth_y - 1.0 and abs(v[i, 0]) < 0.6, i


def test_spec_shapes():
    spec = PatchSpec()
    assert spec.num_patches == 32
    assert len(set(spec.landmark_ids.tolist())) == 32
    assert spec.region_ids.shape == (32,)


def test_view_angles_frontal_face(canonical_vertices):
    from pulsar.preprocess.patches import landmarks_to_pixels

    lm = canonical_landmarks_norm(canonical_vertices, 640, 480)
    ang = patch_view_angles(landmarks_to_pixels(lm, 640, 480), PatchSpec())
    spec = PatchSpec()
    names = list(spec.regions)
    reg = np.array([names[r] for r in spec.region_ids])
    assert np.all(ang[reg == "forehead"] < 45)
    assert np.all(ang < 75)
