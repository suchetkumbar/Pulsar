import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pulsar.data.loading import synthetic_windows  # noqa: E402
from pulsar.models.features import TraceFeatures  # noqa: E402
from pulsar.models.losses import neg_pearson_lagged, patch_snr_db  # noqa: E402
from pulsar.models.tracenet import DFTHead, TraceNet, TraceNetConfig, hr_from_logits  # noqa: E402
from pulsar.preprocess.patches import PatchSpec  # noqa: E402
from pulsar.signal.spectral import hr_grid  # noqa: E402
from pulsar.train.dataset import WindowDataset  # noqa: E402
from pulsar.train.loop import TrainConfig, train  # noqa: E402

FS = 30.0
REG = PatchSpec().region_ids


@pytest.fixture(scope="module")
def ws():
    return synthetic_windows(n_subjects=6, videos_per_subject=1, duration_s=30, stride_s=4, seed=0)


def _batch(ws, n=8):
    ds = WindowDataset(ws)
    items = [ds[i] for i in range(n)]
    return {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in items[0]}


def test_in_graph_pos_feature_recovers_hr(ws):
    b = _batch(ws)
    feats, desc, mask, pos_raw = TraceFeatures(REG)(b["x"], b["valid"], b["n_pix"], b["motion"])
    assert feats.shape == (8, 32, 5, 300) and desc.shape[:2] == (8, 32) and mask.all()
    face_pos = pos_raw.mean(1).numpy()
    errs = [abs(hr_grid(face_pos[i], FS) - float(b["hr"][i])) for i in range(8)]
    assert np.mean(errs) < 3.0


def test_dft_head_peak_matches_numpy():
    bpm = np.arange(40, 201, 1.0)
    head = DFTHead(300, FS, bpm)
    t = np.arange(300) / FS
    x = torch.tensor(np.sin(2 * np.pi * 1.3 * t), dtype=torch.float32)[None]
    peak = bpm[int(head(x).argmax())]
    assert abs(peak - 78.0) <= 1.0


def test_tracenet_shapes_and_masking(ws):
    b = _batch(ws, 4)
    model = TraceNet(TraceNetConfig(), REG).eval()
    out = model(b["x"], b["valid"], b["n_pix"], b["motion"])
    assert out["bvp"].shape == (4, 300) and out["hr_logits"].shape == (4, 161)
    np.testing.assert_allclose(out["alpha"].sum(1).detach().numpy(), 1.0, atol=1e-5)
    v = b["valid"].clone()
    v[:, :16] = False
    out2 = model(b["x"], v, b["n_pix"], b["motion"])
    assert out2["alpha"][:, :16].max().item() < 1e-6  # invalid patches get no weight
    assert model.num_params() < 300_000


def test_hr_from_logits_confidence():
    bpm = torch.arange(40, 201, 1.0)
    peaked = -((bpm - 70) ** 2)[None] / 2.0
    flat = torch.zeros(1, bpm.numel())
    hr, mode, conf = hr_from_logits(torch.cat([peaked, flat]), bpm)
    assert float(mode[0]) == 70 and float(conf[0]) > 0.99 and float(conf[1]) < 0.1


def test_losses():
    t = torch.arange(300) / FS
    y = torch.sin(2 * np.pi * 1.2 * t)[None]
    assert float(neg_pearson_lagged(y, torch.roll(y, 4, -1))) < 0.02  # tolerates lag
    snr = patch_snr_db(torch.stack([y, y + 3 * torch.randn_like(y)], 1), torch.tensor([72.0]), FS)
    assert float(snr[0, 0]) > float(snr[0, 1])


def test_training_smoke_reduces_error(ws, tmp_path):
    tr, va = ws[:5], ws[5:]
    _, log, best = train(
        tr,
        va,
        TraceNetConfig(width=16, enc_blocks=2, dec_blocks=1),
        TrainConfig(epochs=10, batch=8, lr=3e-3, device="cpu", warmup_epochs=1),
        None,
        REG,
        "smoke",
        runs_root=tmp_path,
    )
    assert (log.dir / "metrics.csv").exists() and (log.dir / "best.pt").exists()
    assert best["mae"] < 15.0
