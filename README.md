# Pulsar

**Skin-tone-fair, on-device camera heart-rate estimation (remote photoplethysmography).**

> ⚠️ **Research prototype — not a medical device.** Do not use Pulsar's outputs for diagnosis, treatment, or any health decision.

Pulsar estimates heart rate from ordinary RGB face video, entirely on-device. It is built to stay accurate for darker skin tones and under real-world indoor lighting (fluorescent flicker, ceiling fans, window light).

*Work in progress. Results, figures, and a demo will appear here as milestones complete.*

## Quick start (development)

```bash
uv sync                     # core + dev tools + face landmarks (MediaPipe, OpenCV)
uv sync --group ml          # + CUDA PyTorch for training
uv run pytest               # synthetic-data test suite (no datasets needed)
```

Live demo on your own webcam (classical POS until a model is trained):

```bash
uv run python -m pulsar.demo.webcam
```

## Datasets
Pulsar uses public research datasets (UBFC-rPPG, PURE, MMPD) under their own licences. **No dataset content is distributed with this repository.** Request access from each dataset's authors.

## License
MIT (code). Dataset licences apply separately.
