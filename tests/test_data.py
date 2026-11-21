import json

import numpy as np
import pytest

from pulsar.data.bundle import TraceBundle
from pulsar.data.gt import apply_alignment, estimate_alignment
from pulsar.data.readers import PUREReader, UBFCReader
from pulsar.data.synthetic import synthetic_bundle
from pulsar.data.windows import make_windows
from pulsar.signal.classical import pos
from pulsar.signal.filters import postprocess_bvp
from pulsar.signal.spectral import hr_grid, snr_db

FS = 30.0


def test_bundle_roundtrip(tmp_path):
    b = synthetic_bundle(duration_s=5, seed=3)
    b.lag_s = 0.2
    b.meta["fitzpatrick"] = np.int64(4)
    p = b.save(tmp_path / "x.npz")
    c = TraceBundle.load(p)
    np.testing.assert_array_equal(c.rgb_lin, b.rgb_lin)
    np.testing.assert_array_equal(c.valid, b.valid)
    assert c.gt_spo2 is None and c.gt_hr is not None
    assert c.lag_s == pytest.approx(0.2) and c.meta["fitzpatrick"] == 4


@pytest.mark.parametrize("gt_sign", [1, -1])
def test_alignment_recovers_lag_and_orients_sign(gt_sign):
    b = synthetic_bundle(duration_s=30, lag_s=0.2, gt_sign=gt_sign, seed=1)
    al = estimate_alignment(b)
    assert al.lag_s == pytest.approx(0.2, abs=2 / FS)
    assert al.corr > 0.8
    ref = synthetic_bundle(duration_s=30, lag_s=0.2, gt_sign=1, seed=1)
    assert al.sign * gt_sign == estimate_alignment(ref).sign  # orientation is consistent


def test_windows_targets_and_pos_accuracy():
    b = synthetic_bundle(duration_s=40, hr_bpm=80, seed=2, jitter_ms=3, drop_frac=0.02)
    apply_alignment(b, estimate_alignment(b))
    w = make_windows(b, FS, win_s=10, stride_s=2)
    assert len(w) >= 10
    assert w.x.shape[1:] == (300, 32, 3)
    assert np.all(np.abs(w.hr - 80) < 10)  # GT HR drifts slowly around 80
    rgb = w.face_mean_rgb()
    errs = [abs(hr_grid(postprocess_bvp(pos(rgb[i], FS), FS), FS) - w.hr[i]) for i in range(len(w))]
    assert np.mean(errs) < 2.0


def test_darker_synthetic_skin_has_lower_snr():
    """Same noise model, less light reflected -> lower SNR (motivates PhysTone-Aug)."""
    out = {}
    for skin in ("light", "dark"):
        b = synthetic_bundle(duration_s=30, skin=skin, seed=5, noise_a=5e-3, n_pix=100)
        apply_alignment(b, estimate_alignment(b))
        w = make_windows(b, FS, 10, 5)
        rgb = w.face_mean_rgb()
        out[skin] = np.mean(
            [snr_db(postprocess_bvp(pos(rgb[i], FS), FS), w.hr[i], FS) for i in range(len(w))]
        )
    assert out["light"] > out["dark"] + 2.0


# ------------------------------------------------------------------ readers on tiny fixtures
def test_ubfc_reader_gt(tmp_path):
    d = tmp_path / "subject3"
    d.mkdir()
    t = np.arange(10) / 30
    (d / "ground_truth.txt").write_text(
        " ".join(map(str, np.sin(t))) + "\n" + " ".join(["70"] * 10) + "\n" + " ".join(map(str, t + 5)) + "\n"
    )
    (d / "vid.avi").write_bytes(b"")
    r = UBFCReader(tmp_path)
    recs = r.list_videos()
    assert [x.video_id for x in recs] == ["subject3"]
    gt = r.load_gt(recs[0])
    np.testing.assert_allclose(gt.t, t)  # re-zeroed
    assert gt.hr.shape == (10,)


def test_pure_reader_gt(tmp_path):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path / "01-02"
    (d / "01-02").mkdir(parents=True)
    t0 = 1_000_000_000
    for k in range(3):
        cv2.imwrite(str(d / "01-02" / f"Image{t0 + k * 33_333_333}.png"), np.zeros((4, 4, 3), np.uint8))
    pk = [
        {"Timestamp": t0 + k * 16_666_667, "Value": {"waveform": k, "pulseRate": 70, "o2saturation": 98}}
        for k in range(6)
    ]
    (d / "01-02.json").write_text(json.dumps({"/FullPackage": pk, "/Image": []}))
    r = PUREReader(tmp_path)
    rec = r.list_videos()[0]
    assert rec.meta["activity"] == "talking"
    gt = r.load_gt(rec)
    np.testing.assert_allclose(gt.t[:2], [0, 0.016666667], atol=1e-6)
    assert np.all(gt.spo2 == 98)
    frames = list(r.iter_frames(rec))
    assert len(frames) == 3 and frames[1][0] == pytest.approx(0.0333, abs=1e-3)
