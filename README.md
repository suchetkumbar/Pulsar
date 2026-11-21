# Pulsar

**Skin-tone-fair, on-device camera heart-rate estimation (remote photoplethysmography).**

> ⚠️ **Research prototype — not a medical device.** Do not use Pulsar's outputs for diagnosis, treatment, or any health decision.

Pulsar estimates heart rate from ordinary RGB face video, entirely on-device. It is built to stay accurate for darker skin tones and under real-world indoor lighting (fluorescent flicker, ceiling fans, window light).

*Work in progress. Results, figures, and a demo will appear here as milestones complete.*

## Quick start (development)

```bash
uv sync                     # core + dev tools + face landmarks (MediaPipe, OpenCV)
uv sync --group ml          # + CUDA PyTorch for training
uv sync --group export      # + ONNX / onnxruntime for on-device export
uv run pytest               # synthetic-data test suite (no datasets needed)
```

On slow or per-connection-throttled networks, fetch big wheels with `scripts/pget.sh` (parallel, resumable,
SHA-256-verified against `uv.lock`), install them with `uv pip install --no-deps <wheel>`, and use
`uv run --no-sync`.

Export a trained model for the phone (ONNX + parity check + golden test fixtures):

```bash
uv run python scripts/export_model.py --model runs/<run>/best.pt
```

Live demo on your own webcam (classical POS until a model is trained):

```bash
uv run python -m pulsar.demo.webcam
```

## Android

`android/` contains the phone app:

- `core/` is a pure-Kotlin port of the live pipeline. Its tests check it against the Python reference on the
  desktop, and its patch means are bit-identical to Python's.
- `app/` is CameraX + MediaPipe + ONNX Runtime Mobile, and it has no network permission.

See [android/README.md](android/README.md). The core tests need only a JDK:

```bash
bash scripts/android_core_tests.sh fetch   # once: Kotlin compiler + test jars from Maven Central
bash scripts/android_core_tests.sh
```

## Datasets
Pulsar uses public research datasets (UBFC-rPPG, PURE, MMPD) under their own licences. **No dataset content is distributed with this repository.** Request access from each dataset's authors.

## License
MIT (code). Dataset licences apply separately.
