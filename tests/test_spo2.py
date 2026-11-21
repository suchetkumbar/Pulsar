import numpy as np
import pandas as pd

from pulsar.data.gt import apply_alignment, estimate_alignment
from pulsar.data.loading import bundle_to_windows
from pulsar.data.synthetic import PBV, synthetic_bundle
from pulsar.data.windows import window_spo2
from pulsar.vitals.spo2 import LinearCalibration, evaluate_loso, loso_predict, window_features, window_rr

FS = 30.0


def _windows(spo2, seed, **kw):
    b = synthetic_bundle(duration_s=30.0, spo2=spo2, seed=seed, **kw)
    apply_alignment(b, estimate_alignment(b))
    return bundle_to_windows(b, FS, 10.0, 2.0)


def test_window_spo2_target_and_dropout():
    b = synthetic_bundle(duration_s=20.0, spo2=95.0, seed=1)
    assert window_spo2(b, 2.0, 12.0) == 95.0
    b.gt_spo2[(b.gt_t > 5) & (b.gt_t < 6)] = 127.0  # CMS50E-style dropout code
    assert np.isnan(window_spo2(b, 2.0, 12.0))
    assert window_spo2(b, 8.0, 18.0) == 95.0
    w = _windows(None, 0)
    assert w.spo2 is None


def test_rr_recovers_pbv_ratio():
    w = _windows(97.0, 3, noise_a=2e-4)
    df = window_features([w])
    expected = PBV[0] / PBV[2]  # at spo2 == spo2_ref the toy model leaves PBV unchanged
    assert np.allclose(df["rr_rb"], expected, rtol=0.05), df["rr_rb"].describe()
    assert (df["q"] > 0.5).all()


def test_rr_rejects_invalid_patches():
    w = _windows(97.0, 4)
    x, v = w.x[0].copy(), w.valid[0].copy()
    v[:, :5] = False
    r = window_rr(x, v, w.n_pix[0], np.sin(np.linspace(0, 20, x.shape[0])), FS)
    assert np.isnan(r["rr_patch"][:5]).all()
    flat = window_rr(x, v, w.n_pix[0], np.zeros(x.shape[0]), FS)
    assert np.isnan(flat["rr"])


def test_linear_calibration_and_loso_beats_constant():
    rng = np.random.default_rng(0)
    ws = []
    for s, sat in enumerate(rng.uniform(88, 99, 8)):
        w = _windows(float(sat), 100 + s)
        w.subject = f"s{s}"
        ws.append(w)
    df = window_features(ws)
    cal = LinearCalibration().fit(df["rr_rb"], df["spo2_true"])
    assert cal.b < 0  # more red pulsatility -> lower saturation
    res = evaluate_loso(df, ("rr_rb",))
    assert res.loc["RoR RB", "subject_mae"] < 0.5 * res.loc["constant-mean", "subject_mae"]
    assert res.loc["RoR RB", "pearson"] > 0.9


def test_loso_never_sees_held_out_subject():
    df = pd.DataFrame({"subject": ["a", "a", "b", "b"], "spo2_true": [90.0, 90.0, 99.0, 99.0], "f": 1.0})
    # constant-mean of the *other* subject only
    assert np.allclose(loso_predict(df, None), [99, 99, 90, 90])
