from dataclasses import replace

import numpy as np
import pytest

from pulsar.aug import illumsim, phystone
from pulsar.data.gt import apply_alignment, estimate_alignment
from pulsar.data.synthetic import synthetic_bundle
from pulsar.data.windows import make_windows
from pulsar.signal.alias import alias_frequency
from pulsar.signal.classical import pos
from pulsar.signal.filters import postprocess_bvp
from pulsar.signal.spectral import snr_db

FS = 30.0
OFF = illumsim.IllumSimConfig(p_flicker=0, p_fan=0, p_window=0, p_cct=0, p_ae=0, p_fps=0)


def _window(seed=0, skin="light", n_pix=400):
    b = synthetic_bundle(duration_s=24, skin=skin, seed=seed, n_pix=n_pix, noise_a=1e-3)
    apply_alignment(b, estimate_alignment(b))
    w = make_windows(b, FS, 10, 10)
    return w


def _peak_hz(x, fs, lo=0.3, hi=6.0):
    f = np.fft.rfftfreq(x.size, 1 / fs)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(p[m])]


def _pos_snr(x, hr):
    rgb = np.nanmean(x, axis=1)
    return snr_db(postprocess_bvp(pos(rgb, FS), FS), hr, FS)


# ------------------------------------------------------------------ PhysTone-Aug (C1)
def test_darken_identity_and_affine_commutes_with_averaging(rng):
    x = rng.uniform(0.1, 0.6, (50, 4, 3))
    sig = phystone.DEFAULT_SIGMA_MELANIN
    np.testing.assert_allclose(phystone.darken(x, 0.0, sig, 0.1)[0], x)
    # pixel-level then average == average then trace-level (exactness at trace level)
    pix = rng.uniform(0.1, 0.6, (50, 100, 3))
    dc_pix = pix.mean(axis=(0, 1), keepdims=True)
    a = phystone.darken(pix, 0.7, sig, 0.1, dc=dc_pix)[0].mean(axis=1)
    b = phystone.darken(pix.mean(axis=1, keepdims=True), 0.7, sig, 0.1, dc=dc_pix)[0][:, 0]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_relative_pulse_factor_physics():
    sig = phystone.DEFAULT_SIGMA_MELANIN
    f = [phystone.relative_pulse_factor(d, sig, 0.1) for d in (0.0, 0.5, 1.0, 2.0)]
    assert np.allclose(f[0], 1.0)
    assert all(np.all(f[i + 1] < f[i]) for i in range(3))  # monotone in melanin
    assert np.all(f[-1][2] < f[-1][0])  # blue loses more than red
    assert np.allclose(phystone.relative_pulse_factor(2.0, sig, 0.0), 1.0)  # s=0: no AC/DC loss


def test_augment_hits_target_ita_and_lowers_snr(rng):
    w = _window()
    x, hr = w.x[0], w.hr[0]
    xa, info = phystone.augment(x, w.n_pix[0], rng, target_ita=-20.0)
    assert info["ita_to"] == pytest.approx(-20.0, abs=1.5)
    assert info["ita_from"] > 30
    assert _pos_snr(xa, hr) < _pos_snr(x, hr) - 3.0


def test_naive_tone_darkens_without_snr_loss(rng):
    w = _window()
    x, hr = w.x[0], w.hr[0]
    xn = phystone.naive_tone(x, rng, strength=(0.8, 0.8))
    assert xn.mean() < x.mean() * 0.7
    assert abs(_pos_snr(xn, hr) - _pos_snr(x, hr)) < 1.0  # the gap PhysTone-Aug fixes


def test_colour_only_ablation_keeps_relative_pulse(rng):
    w = _window()
    cfg = phystone.PhysToneConfig(surface_frac=0.0, noise_topup=False)
    xa, _ = phystone.augment(w.x[0], w.n_pix[0], rng, cfg, target_ita=-20.0)
    rel0 = w.x[0].std(axis=0) / w.x[0].mean(axis=0)
    rel1 = xa.std(axis=0) / xa.mean(axis=0)
    np.testing.assert_allclose(rel1, rel0, rtol=1e-4)


# ------------------------------------------------------------------ IllumSim-IN (C2)
def _flat(t_len=600, p=4, level=0.3):
    x = np.full((t_len, p, 3), level, np.float64)
    center = np.tile([[0.5, 0.4]], (p, 1))
    return x, center, np.full(p, 10_000)


def test_all_off_is_identity(rng):
    w = _window()
    out, info = illumsim.simulate(w.x[0], np.full((32, 2), 0.5), w.n_pix[0], FS, rng, OFF)
    np.testing.assert_allclose(out, w.x[0], atol=1e-7)
    assert info.components == []


@pytest.mark.parametrize("fps", [24.5, 30.0])
def test_mains_flicker_aliases_as_predicted(fps):
    rng = np.random.default_rng(1)
    x, c, n = _flat()
    cfg = replace(
        OFF,
        p_flicker=1,
        p_fps=1,
        capture_fps=(fps,),
        jitter_ms=(0, 0),
        drop_frac=(0, 0),
        flicker_depth=(0.5, 0.5),
        exposures_s=(1 / 200,),
        grid_drift=0.0,
        noise_a=0,
        noise_b=0,
    )
    out, info = illumsim.simulate(x, c, n, FS, rng, cfg)
    expected = alias_frequency(100.0, fps)
    sig = out[:, 0, 1]
    if expected <= 6.0:
        assert _peak_hz(sig, FS) == pytest.approx(expected, abs=0.1)
    assert sig.std() > 0.01  # flicker visible


def test_antibanding_exposure_cancels_flicker():
    rng = np.random.default_rng(2)
    x, c, n = _flat()
    cfg = replace(
        OFF,
        p_flicker=1,
        flicker_depth=(0.5, 0.5),
        exposures_s=(1 / 100,),
        grid_drift=0.0,
        noise_a=0,
        noise_b=0,
    )
    out, info = illumsim.simulate(x, c, n, FS, rng, cfg)
    assert info.params["flicker_att"] < 1e-12
    np.testing.assert_allclose(out, x, atol=1e-9)


def test_fan_harmonic_aliases_into_hr_band():
    rng = np.random.default_rng(3)
    x, c, n = _flat()
    cfg = replace(
        OFF,
        p_fan=1,
        p_fps=1,
        capture_fps=(20.0,),
        jitter_ms=(0, 0),
        drop_frac=(0, 0),
        fan_depth=(0.1, 0.1),
        fan_rpm=(350.0, 350.0),
        fan_blades=(3,),
        fan_chirp_prob=0.0,
        noise_a=0,
        noise_b=0,
    )
    out, info = illumsim.simulate(x, c, n, FS, rng, cfg)
    assert info.params["fan_bpf_hz"] == pytest.approx(17.5)
    assert _peak_hz(out[:, 0, 1], FS) == pytest.approx(alias_frequency(17.5, 20.0), abs=0.1)  # 2.5 Hz


def test_severity_degrades_pos_snr():
    w = _window(seed=4)
    x, hr, n_pix = w.x[0], w.hr[0], w.n_pix[0]
    center = np.tile([[0.5, 0.5]], (x.shape[1], 1))
    snr = {}
    for s in (0, 2, 4):
        vals = []
        for k in range(6):
            out, _ = illumsim.simulate(
                x, center, n_pix, FS, np.random.default_rng(k), illumsim.severity_preset(s)
            )
            vals.append(_pos_snr(out, hr))
        snr[s] = np.mean(vals)
    assert snr[0] > snr[2] > snr[4]


def test_output_shape_and_finite(rng):
    w = _window()
    cfg = illumsim.IllumSimConfig(p_flicker=1, p_fan=1, p_window=1, p_cct=1, p_ae=1, p_fps=1)
    out, info = illumsim.simulate(w.x[0], np.full((32, 2), 0.5), w.n_pix[0], FS, rng, cfg)
    assert out.shape == w.x[0].shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert set(info.components) == {"fps", "flicker", "fan", "window", "cct", "ae"}
