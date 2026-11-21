import numpy as np
import pytest
from scipy import sparse

from pulsar.signal import alias, classical, filters, spectral

FS = 30.0


def synth_rgb(hr_bpm=75.0, dur=20.0, noise=0.0005, drift=0.03, drift_hz=0.07, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    p = np.sin(2 * np.pi * hr_bpm / 60 * t) + 0.4 * np.sin(4 * np.pi * hr_bpm / 60 * t - 0.8)
    pbv = np.array([0.33, 0.77, 0.53])
    dc = np.array([0.5, 0.33, 0.25])
    illum = 1 + drift * np.sin(2 * np.pi * drift_hz * t)
    rgb = dc * illum[:, None] * (1 + 0.003 * pbv * p[:, None])
    return rgb + rng.normal(0, noise, rgb.shape) * dc


@pytest.mark.parametrize("method", ["POS", "CHROM", "GREEN", "LGI", "ICA"])
def test_methods_recover_hr(method):
    rgb = synth_rgb(hr_bpm=75.0)
    bvp = filters.postprocess_bvp(classical.run_method(method, rgb, FS), FS)
    assert abs(spectral.hr_grid(bvp, FS) - 75.0) < 2.0


def test_pos_cancels_inband_intensity_modulation():
    """Common-mode brightness modulation inside the HR band (e.g. aliased flicker) defeats GREEN
    but is projected out by POS (it lies along the intensity direction)."""
    rgb = synth_rgb(hr_bpm=66.0, drift=0.01, drift_hz=1.7, noise=0.0002)
    s_pos = spectral.snr_db(filters.postprocess_bvp(classical.pos(rgb, FS), FS), 66.0, FS)
    s_g = spectral.snr_db(filters.postprocess_bvp(classical.green(rgb), FS), 66.0, FS)
    assert s_pos > s_g


def test_detrend_matches_dense_toolbox_formula():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n) + np.linspace(0, 5, n) ** 2
    ones = np.ones(n)
    d = sparse.spdiags(np.array([ones, -2 * ones, ones]), np.array([0, 1, 2]), n - 2, n).toarray()
    h = np.identity(n)
    dense = (h - np.linalg.inv(h + 100.0**2 * d.T @ d)) @ x
    np.testing.assert_allclose(filters.detrend_tarvainen(x, 100.0), dense, atol=1e-8)


def test_bandpass_attenuates_out_of_band():
    t = np.arange(600) / FS
    inband = np.sin(2 * np.pi * 1.2 * t)
    low = np.sin(2 * np.pi * 0.1 * t)
    high = np.sin(2 * np.pi * 8.0 * t)
    y = filters.bandpass(inband + low + high, FS)
    assert np.corrcoef(y, inband)[0, 1] > 0.98


def test_hr_grid_refines_between_bins():
    t = np.arange(300) / FS
    x = np.sin(2 * np.pi * (73.4 / 60) * t)
    assert abs(spectral.hr_grid(x, FS) - 73.4) < 0.3


def test_toolbox_fft_hr_bin_resolution():
    t = np.arange(900) / FS
    x = np.sin(2 * np.pi * 1.25 * t)  # 75 bpm
    # nfft=1024 -> 1.76 bpm bins; peak must be within one bin
    assert abs(spectral.hr_fft_toolbox(x, FS) - 75.0) < 1.8


def test_snr_monotonic_in_noise():
    rng = np.random.default_rng(0)
    t = np.arange(300) / FS
    clean = np.sin(2 * np.pi * 1.2 * t)
    snrs = [spectral.snr_db(clean + s * rng.normal(size=t.size), 72.0, FS) for s in (0.1, 0.5, 2.0)]
    assert snrs[0] > snrs[1] > snrs[2]
    tb = [spectral.snr_db_toolbox(clean + s * rng.normal(size=t.size), 72.0, FS) for s in (0.1, 2.0)]
    assert tb[0] > tb[1]


def test_alias_frequencies_match_docs_table():
    assert alias.alias_frequency(100.0, 30.0) == pytest.approx(10.0)
    assert alias.alias_frequency(100.0, 25.0) == pytest.approx(0.0)
    assert alias.alias_frequency(100.0, 24.5) == pytest.approx(2.0)
    assert alias.alias_frequency(52.5, 25.0) == pytest.approx(2.5)
    assert alias.alias_frequency(17.5, 20.0) == pytest.approx(2.5)
    assert alias.fan_blade_pass_hz(350, 3) == pytest.approx(17.5)


def test_alias_matches_sampled_spectrum():
    """A 100 Hz flicker sampled at 24.5 fps really shows up at 2.0 Hz."""
    fs = 24.5
    t = np.arange(int(60 * fs)) / fs
    x = np.cos(2 * np.pi * 100.0 * t)
    f = np.fft.rfftfreq(t.size, 1 / fs)
    peak = f[np.argmax(np.abs(np.fft.rfft(x)))]
    assert peak == pytest.approx(alias.alias_frequency(100.0, fs), abs=0.05)


def test_rgb_shape_validation():
    with pytest.raises(ValueError):
        classical.pos(np.zeros((10, 4)), FS)
