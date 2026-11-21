"""Export TraceNet for on-device inference (docs 07 §2): ONNX (+ optional static INT8), parity checks,
and a metadata JSON that is the single source of truth for the Android pipeline.

The exported graph takes the same inputs as training, all float32 so every runtime can feed them:
    x [1, P, T, 3] linear RGB patch means on the 30 Hz grid, valid [1, P, T] (0/1),
    n_pix [1, P] skin pixel counts, motion [1] landmark motion.
Outputs: bvp [1, T], hr [1] (soft-argmax bpm), hr_mode [1], conf [1] (calibrated probability mass
within +-5 bpm of the mode), alpha [1, P] patch attention, hr_logits [1, K].
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pulsar.models.tracenet import TraceNet, hr_from_logits
from pulsar.preprocess.live import window_contract
from pulsar.preprocess.patches import PatchSpec
from pulsar.preprocess.traces import extractor_contract

INPUT_NAMES = ["x", "valid", "n_pix", "motion"]
OUTPUT_NAMES = ["bvp", "hr", "hr_mode", "conf", "alpha", "hr_logits"]


class ExportWrapper(nn.Module):
    def __init__(self, model: TraceNet, temperature: float | None = None, tol_bpm: float = 5.0):
        super().__init__()
        self.model = model.eval()
        self.temperature = float(getattr(model, "temperature", 1.0) if temperature is None else temperature)
        self.tol_bpm = tol_bpm

    def forward(self, x, valid, n_pix, motion):
        out = self.model(x, valid > 0.5, n_pix, motion)
        hr, hr_mode, conf = hr_from_logits(out["hr_logits"], self.model.bpm, self.temperature, self.tol_bpm)
        return out["bvp"], hr, hr_mode, conf, out["alpha"], out["hr_logits"]


def sample_inputs(model: TraceNet, batch: int = 1, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Plausible random inputs (skin-like DC + small noise) with the model's static shapes."""
    g = torch.Generator().manual_seed(seed)
    p, t = model.features.region_onehot.shape[0], model.cfg.t_len
    dc = torch.tensor([0.35, 0.22, 0.15]).view(1, 1, 1, 3)
    x = dc * (1 + 0.01 * torch.randn(batch, p, t, 3, generator=g))
    valid = torch.ones(batch, p, t)
    valid[:, : max(1, p // 8)] = 0.0  # a few fully invalid patches exercise the attention mask
    n_pix = torch.full((batch, p), 200.0)
    motion = torch.zeros(batch)
    return x, valid, n_pix, motion


def export_onnx(
    model: TraceNet,
    path: str | Path,
    opset: int = 17,
    temperature: float | None = None,
    exporter: str = "auto",
) -> dict:
    """Export with static batch 1. exporter: "torchscript", "dynamo" (opset >= 18) or "auto" (try both)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model.cpu(), temperature).eval()
    args = sample_inputs(model)
    order = {"auto": (False, True), "torchscript": (False,), "dynamo": (True,)}[exporter]
    errors = []
    for dynamo in order:
        ops = max(opset, 18) if dynamo else opset
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    wrapper,
                    args,
                    str(path),
                    input_names=INPUT_NAMES,
                    output_names=OUTPUT_NAMES,
                    opset_version=ops,
                    dynamo=dynamo,
                    do_constant_folding=True,
                )
            return {"path": str(path), "exporter": "dynamo" if dynamo else "torchscript", "opset": ops}
        except Exception as e:  # noqa: BLE001 - fall through to the other exporter, report both
            errors.append(f"{'dynamo' if dynamo else 'torchscript'}: {type(e).__name__}: {e}")
    raise RuntimeError("ONNX export failed:\n" + "\n".join(errors))


def onnx_session(path: str | Path, threads: int = 0):
    """CPU session; threads=0 lets onnxruntime choose (set 1-2 to mimic a phone's budget)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_onnx(path_or_session, inputs: tuple[torch.Tensor, ...]) -> list[np.ndarray]:
    sess = onnx_session(path_or_session) if isinstance(path_or_session, (str, Path)) else path_or_session
    feed = {n: t.numpy().astype(np.float32) for n, t in zip(INPUT_NAMES, inputs, strict=True)}
    return sess.run(OUTPUT_NAMES, feed)


@torch.no_grad()
def parity(model: TraceNet, path: str | Path, inputs: tuple[torch.Tensor, ...] | None = None) -> dict:
    """Max |torch - onnxruntime| per output, plus the HR difference in bpm."""
    inputs = sample_inputs(model, seed=1) if inputs is None else inputs
    ref = [o.numpy() for o in ExportWrapper(model.cpu()).eval()(*inputs)]
    got = run_onnx(path, inputs)
    return {n: float(np.abs(r - g).max()) for n, r, g in zip(OUTPUT_NAMES, ref, got, strict=True)}


def quantize_int8(path_in: str | Path, path_out: str | Path, calib: list[tuple[torch.Tensor, ...]]) -> Path:
    """Static QDQ INT8 (per-channel weights) calibrated on real windows; the DFT head stays float."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(
                [
                    {n: t.numpy().astype(np.float32) for n, t in zip(INPUT_NAMES, c, strict=True)}
                    for c in calib
                ]
            )

        def get_next(self):
            return next(self.it, None)

    path_out = Path(path_out)
    pre = path_out.with_suffix(".pre.onnx")
    quant_pre_process(str(path_in), str(pre))
    quantize_static(
        str(pre),
        str(path_out),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv"],
    )
    pre.unlink(missing_ok=True)
    return path_out


def model_metadata(model: TraceNet, spec: PatchSpec | None = None) -> dict:
    """Everything the phone needs to reproduce training-time inputs exactly."""
    spec = spec or PatchSpec()
    region_ids = model.features.region_onehot.argmax(-1).cpu().numpy()
    if not np.array_equal(region_ids, spec.region_ids):
        raise ValueError("model region layout does not match the PatchSpec; pass the spec used for training")
    return {
        "format": "pulsar-tracenet",
        "inputs": {
            "x": "float32 [1, P, T, 3] linear-RGB patch means (inverse sRGB before averaging)",
            "valid": "float32 [1, P, T] 1 = patch valid at that grid sample",
            "n_pix": "float32 [1, P] mean skin-pixel count per patch over the window",
            "motion": "float32 [1] mean landmark motion (face-width units / frame)",
        },
        "outputs": OUTPUT_NAMES,
        "fs_hz": model.cfg.fs,
        "t_len": model.cfg.t_len,
        "num_patches": int(region_ids.size),
        "bpm_grid": [model.cfg.bpm_lo, model.cfg.bpm_hi, 1.0],
        "temperature": float(getattr(model, "temperature", 1.0)),
        "patch_spec": spec.contract(),
        "extractor": extractor_contract(),
        "window": window_contract(model.cfg.fs, model.cfg.t_len / model.cfg.fs),
        "num_params": model.num_params(),
    }


def write_metadata(model: TraceNet, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_metadata(model) | (extra or {}), indent=2), encoding="utf-8")
    return path
