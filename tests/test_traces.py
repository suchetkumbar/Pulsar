import numpy as np
import pytest

from pulsar.preprocess.color import srgb_to_linear
from pulsar.preprocess.patches import PatchSpec, landmarks_to_pixels, patch_boxes
from pulsar.preprocess.traces import TraceExtractor, extract_video

from .conftest import canonical_landmarks_norm

W, H = 640, 480
SKIN_SRGB = np.array([200, 150, 120], np.uint8)


class FixedLandmarker:
    def __init__(self, lm, drop_every: int = 0):
        self.lm, self.drop_every, self.n = lm, drop_every, 0

    def detect(self, frame_rgb, timestamp_ms):
        self.n += 1
        if self.drop_every and self.n % self.drop_every == 0:
            return None
        return self.lm


def _face_image(lm_norm):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 30, 30)  # background
    px = landmarks_to_pixels(lm_norm, W, H)
    x0, y0 = px[:, :2].min(0).astype(int)
    x1, y1 = px[:, :2].max(0).astype(int)
    img[y0:y1, x0:x1] = SKIN_SRGB
    return img


def test_patch_means_equal_uniform_skin(canonical_vertices):
    lm = canonical_landmarks_norm(canonical_vertices, W, H)
    img = _face_image(lm)
    ex = TraceExtractor()
    st = ex.process(img, lm)
    assert st.face_found and st.valid.all()
    expected = srgb_to_linear(SKIN_SRGB / 255.0)
    np.testing.assert_allclose(st.rgb_lin, np.broadcast_to(expected, st.rgb_lin.shape), atol=1e-5)
    np.testing.assert_allclose(st.rgb_srgb, np.broadcast_to(SKIN_SRGB / 255.0, st.rgb_srgb.shape), atol=1e-5)
    assert (st.n_pix >= PatchSpec().min_pixels).all()


def test_specular_and_dark_pixels_excluded(canonical_vertices):
    lm = canonical_landmarks_norm(canonical_vertices, W, H)
    img = _face_image(lm)
    spec = PatchSpec()
    boxes = patch_boxes(landmarks_to_pixels(lm, W, H), spec, W, H)
    x0, y0, x1, y1 = boxes[0]
    img[y0 : y0 + 2, x0:x1] = (255, 255, 255)  # specular stripe in forehead patch 0
    img[y1 - 2 : y1, x0:x1] = (5, 3, 2)  # hair/shadow stripe
    ex = TraceExtractor(spec)
    ex.process(_face_image(lm), lm)  # initialise skin model on a clean frame
    st = ex.process(img, lm)
    expected = srgb_to_linear(SKIN_SRGB / 255.0)
    np.testing.assert_allclose(st.rgb_lin[0], expected, atol=1e-4)


def test_missing_landmarks_carry_then_invalidate(canonical_vertices):
    lm = canonical_landmarks_norm(canonical_vertices, W, H)
    img = _face_image(lm)
    ex = TraceExtractor(carry_frames=2)
    assert ex.process(img, lm).valid.all()
    assert ex.process(img, None).valid.all()  # carried
    assert ex.process(img, None).valid.all()  # carried
    st = ex.process(img, None)  # exceeded carry
    assert not st.valid.any() and not st.face_found


def test_extract_video_bundle(canonical_vertices):
    lm = canonical_landmarks_norm(canonical_vertices, W, H)
    img = _face_image(lm)
    frames = ((i / 30.0, img) for i in range(20))
    b = extract_video(frames, FixedLandmarker(lm, drop_every=7), meta={"video_id": "x"})
    assert b.rgb_lin.shape == (20, 32, 3) and b.valid.shape == (20, 32)
    assert b.face_found.sum() == 20 - 2
    assert np.isfinite(b.ita_patch).all()
    assert b.meta["patch_spec"]["version"] == "v1"


def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        TraceExtractor().process(np.zeros((10, 10, 3), np.float32), None)
