import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from pulsar.export.onnx_export import export_onnx, model_metadata, parity, write_metadata  # noqa: E402
from pulsar.models.tracenet import TraceNet, TraceNetConfig  # noqa: E402
from pulsar.preprocess.patches import PatchSpec  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = TraceNet(TraceNetConfig(width=16, enc_blocks=2, dec_blocks=1), PatchSpec().region_ids).eval()
    m.temperature = 1.7
    return m


@pytest.mark.parametrize("exporter", ["torchscript", "dynamo"])
def test_onnx_parity(model, tmp_path, exporter):
    if exporter == "dynamo":
        pytest.importorskip("onnxscript")
    path = tmp_path / "m.onnx"
    info = export_onnx(model, path, exporter=exporter)
    assert path.exists() and info["exporter"] == exporter
    d = parity(model, path)  # sample inputs include fully-invalid patches (the NaN regression)
    assert d["bvp"] < 1e-3, d
    assert d["hr"] < 0.05 and d["conf"] < 1e-3, d
    assert d["hr_mode"] == 0.0, d


def test_metadata_is_the_patch_contract(model, tmp_path):
    meta = json.loads(write_metadata(model, tmp_path / "meta.json").read_text())
    spec = PatchSpec()
    assert meta["num_patches"] == spec.num_patches == len(meta["patch_spec"]["landmark_ids"])
    assert meta["patch_spec"]["region_ids"] == spec.region_ids.tolist()
    assert meta["temperature"] == pytest.approx(1.7)
    assert meta["bpm_grid"] == [40.0, 200.0, 1.0]
    bad = PatchSpec(regions={"forehead": [151, 108]})
    with pytest.raises(ValueError):
        model_metadata(model, bad)
    assert np.isfinite(meta["fs_hz"])
    # blocks the Android port reads instead of hard-coding (android/core Config.kt)
    assert meta["patch_spec"]["region_names"] == list(spec.regions)
    assert meta["window"]["t_len"] == meta["t_len"] and meta["window"]["fs_hz"] == meta["fs_hz"]
    from pulsar.preprocess.traces import extractor_contract

    assert meta["extractor"] == extractor_contract()


def test_golden_files_roundtrip(model, tmp_path):
    from pulsar.export.golden import CASES, TOLERANCE, check_golden, write_golden

    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path, exporter="torchscript")
    paths = write_golden(onnx_path, tmp_path / "golden")
    assert len(paths) == len(CASES)
    for p in paths:
        case = json.loads(p.read_text())
        assert len(case["expected"]["t_grid"]) == 300
        assert np.isfinite(case["expected"]["outputs"]["bvp"]).all()
        err = check_golden(p, onnx_path)
        assert err["valid"] == 0, (p.name, err)
        for k, tol in TOLERANCE.items():
            if k in err:
                assert err[k] <= tol, (p.name, k, err[k])
    gaps = json.loads((tmp_path / "golden" / "medium_gaps.json").read_text())
    v = np.asarray(gaps["expected"]["valid"])
    assert v[:, 5].sum() == 0 and v[:, 8:16].min() == 0 and v[:, 20].all()  # invalid patch, gap, clean


def test_live_window_matches_training_resampler():
    from pulsar.data.synthetic import synthetic_bundle
    from pulsar.preprocess.live import live_window
    from pulsar.preprocess.resample import resample

    b = synthetic_bundle(duration_s=12.0, jitter_ms=4.0, drop_frac=0.03, seed=5)
    tg, x, v, npx, mot = live_window(b.t, b.rgb_lin, b.valid, b.n_pix, b.motion)
    tg2, x2, v2 = resample(b.t, b.rgb_lin, fs=30.0, valid=b.valid)
    assert x.shape == (300, b.num_patches, 3) and np.allclose(tg, tg2[-300:])
    assert np.allclose(x, np.nan_to_num(x2[-300:]), atol=1e-6) and (v == v2[-300:]).all()
    assert live_window(b.t[:100], b.rgb_lin[:100], b.valid[:100], b.n_pix[:100], b.motion[:100]) is None
