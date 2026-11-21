"""Print the on-disk structure of a downloaded dataset and check the reader's assumptions.

Run this first after every download, then fix any VERIFY assumption in pulsar/data/readers.py:
    uv run python scripts/inspect_dataset.py --dataset ubfc --root data/raw/ubfc
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from pulsar.data.readers import get_reader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ubfc", "pure", "mmpd"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--n", type=int, default=2, help="records to inspect in detail")
    args = ap.parse_args()

    print(f"== top-level entries under {args.root}")
    for p in sorted(args.root.iterdir())[:15]:
        print("  ", p.name, "/" if p.is_dir() else f"{p.stat().st_size / 1e6:.1f} MB")

    reader = get_reader(args.dataset, args.root)
    recs = reader.list_videos()
    subjects = sorted({r.subject for r in recs})
    print(f"== {len(recs)} videos, {len(subjects)} subjects")

    for rec in recs[: args.n]:
        print(
            f"\n-- {rec.video_id} (subject {rec.subject}) paths={ {k: str(v) for k, v in rec.paths.items()} }"
        )
        if args.dataset == "mmpd":
            mat = reader._load(rec)
            for k, v in mat.items():
                if not k.startswith("__"):
                    a = np.asarray(v)
                    print(f"   key {k!r}: shape={a.shape} dtype={a.dtype}")
            print("   meta:", reader.read_meta(rec, mat))
        gt = reader.load_gt(rec)
        dt = np.diff(gt.t)
        print(
            f"   GT: n={gt.ppg.size}, span={gt.t[-1] - gt.t[0]:.2f}s, "
            f"rate~{1 / np.median(dt):.2f} Hz, ppg range=[{gt.ppg.min():.3g}, {gt.ppg.max():.3g}]"
        )
        if gt.spo2 is not None:
            print(f"   SpO2 range: {np.nanmin(gt.spo2):.1f}-{np.nanmax(gt.spo2):.1f}")
        frames = list(itertools.islice(reader.iter_frames(rec), 90))
        ts = np.array([t for t, _ in frames])
        f0 = frames[0][1]
        print(
            f"   frames: first 90 -> shape={f0.shape} dtype={f0.dtype} "
            f"fps~{1 / np.median(np.diff(ts)):.2f}, mean RGB={f0.reshape(-1, 3).mean(0).round(1)}"
        )


if __name__ == "__main__":
    main()
