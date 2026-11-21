"""Render the 32 skin patches on MediaPipe's canonical face (no dataset faces -> publishable).

uv run python scripts/plot_patch_layout.py  ->  reports/figures/patch_layout.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pulsar.preprocess.patches import PatchSpec, landmarks_to_pixels, patch_boxes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 600, 700


def main() -> None:
    v = np.loadtxt(ROOT / "tests/assets/mediapipe_canonical_face_vertices.csv", delimiter=",")
    span = v[:, 0].max() - v[:, 0].min()
    s = 0.8 * W / span
    lm = np.stack(
        [(W / 2 + v[:, 0] * s) / W, (H / 2 - (v[:, 1] - v[:, 1].mean()) * s) / H, -v[:, 2] * s / W], 1
    )
    spec = PatchSpec()
    px = landmarks_to_pixels(lm, W, H)
    boxes = patch_boxes(px, spec, W, H)
    names = list(spec.regions)
    colors = plt.get_cmap("tab10")(np.arange(len(names)))
    fig, ax = plt.subplots(figsize=(5, 5.8))
    ax.scatter(px[:, 0], px[:, 1], s=1.5, c="0.7")
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        r = spec.region_ids[k]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.35, color=colors[r]))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.8, color=colors[r]))
    for r, n in enumerate(names):
        ax.plot([], [], "s", color=colors[r], label=n)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    ax.set_title("Pulsar skin patches (32) on MediaPipe canonical face", fontsize=10)
    out = ROOT / "reports/figures/patch_layout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
