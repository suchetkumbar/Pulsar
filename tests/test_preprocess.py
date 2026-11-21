import numpy as np
import pytest

from pulsar.preprocess import color, resample


def test_srgb_roundtrip_and_lut():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(color.linear_to_srgb(color.srgb_to_linear(x)), x, atol=1e-9)
    lut = color.srgb8_to_linear_lut()
    assert lut.shape == (256,) and lut[0] == 0 and lut[255] == pytest.approx(1.0)
    assert np.all(np.diff(lut) > 0)


def test_ita_orders_skin_tones():
    light = color.ita_from_linear_rgb(np.array([0.50, 0.33, 0.25]))
    medium = color.ita_from_linear_rgb(np.array([0.32, 0.19, 0.12]))
    dark = color.ita_from_linear_rgb(np.array([0.12, 0.065, 0.04]))
    assert light > medium > dark
    assert color.ita_band(60) == "very light" and color.ita_band(-45) == "dark"


def test_lab_white_point():
    lab = color.linear_rgb_to_lab(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(lab, [100, 0, 0], atol=0.05)


def test_resample_linear_exact_with_jitter():
    rng = np.random.default_rng(0)
    t = np.sort(np.arange(300) / 30 + rng.normal(0, 0.003, 300))
    x = 2.0 * t + 1.0
    tg, xg, vg = resample.resample(t, x, fs=30.0)
    np.testing.assert_allclose(xg, 2.0 * tg + 1.0, atol=1e-9)
    assert vg.all()


def test_resample_flags_gaps_and_keeps_mask_shape():
    t = np.concatenate([np.arange(0, 3, 1 / 30), np.arange(4, 6, 1 / 30)])  # 1 s gap
    x = np.ones((t.size, 4, 3))
    valid = np.ones((t.size, 4), bool)
    valid[:, 2] = False  # patch 2 never valid
    tg, xg, vg = resample.resample(t, x, fs=30.0, valid=valid)
    assert xg.shape == (tg.size, 4, 3) and vg.shape == (tg.size, 4)
    in_gap = (tg > 2.97) & (tg < 3.999)  # last pre-gap sample 2.967 s, next sample 4.0 s
    assert not vg[in_gap].any()
    assert vg[~in_gap][:, [0, 1, 3]].all()
    assert not vg[:, 2].any()


def test_resample_unsorted_and_duplicate_timestamps():
    t = np.array([0.0, 0.1, 0.1, 0.05, 0.2])
    x = np.array([0.0, 1.0, 1.0, 0.5, 2.0])
    tg, xg, _ = resample.resample(t, x, fs=20.0)
    np.testing.assert_allclose(xg, 10 * tg, atol=1e-9)
