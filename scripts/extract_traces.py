"""Extract per-patch traces for every video of a dataset into data/cache/traces/<dataset>/.

    uv run python scripts/extract_traces.py --dataset ubfc --root data/raw/ubfc --workers 6

Each worker owns one MediaPipe landmarker. Existing outputs are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np


def _process(job: tuple[str, str, dict, str, bool]) -> dict:
    dataset, root, rec_dict, out_dir, overwrite = job
    from pulsar.data.gt import apply_alignment, estimate_alignment
    from pulsar.data.readers import VideoRecord, get_reader
    from pulsar.preprocess.face import MediaPipeLandmarker
    from pulsar.preprocess.traces import extract_video

    rec = VideoRecord(
        rec_dict["dataset"],
        rec_dict["video_id"],
        rec_dict["subject"],
        {k: Path(v) for k, v in rec_dict["paths"].items()},
        rec_dict["meta"],
    )
    out = Path(out_dir) / f"{rec.video_id}.npz"
    if out.exists() and not overwrite:
        return {"video_id": rec.video_id, "status": "skipped"}
    t0 = time.time()
    try:
        reader = get_reader(dataset, root)
        meta = {"dataset": dataset, "video_id": rec.video_id, "subject": rec.subject, **rec.meta}
        if hasattr(reader, "read_meta"):
            meta.update(reader.read_meta(rec))
        gt = reader.load_gt(rec)
        with MediaPipeLandmarker() as lm:
            b = extract_video(reader.iter_frames(rec), lm, meta=meta)
        b.gt_t, b.gt_ppg = gt.t, gt.ppg.astype(np.float32)
        b.gt_hr = None if gt.hr is None else gt.hr.astype(np.float32)
        b.gt_spo2 = None if gt.spo2 is None else gt.spo2.astype(np.float32)
        al = estimate_alignment(b)
        apply_alignment(b, al)
        b.save(out)
        return {
            "video_id": rec.video_id,
            "status": "ok",
            "sec": round(time.time() - t0, 1),
            "frames": b.num_frames,
            "face_frac": float(b.face_found.mean()),
            "valid_frac": float(b.valid.mean()),
            "lag_s": al.lag_s,
            "sign": al.sign,
            "align_corr": round(al.corr, 3),
        }
    except Exception as e:  # report and continue with other videos
        return {
            "video_id": rec.video_id,
            "status": "error",
            "error": repr(e),
            "trace": traceback.format_exc(limit=3),
        }


def main() -> None:
    from pulsar.data.readers import get_reader

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ubfc", "pure", "mmpd"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = args.out or Path("data/cache/traces") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = get_reader(args.dataset, args.root).list_videos()[: args.limit]
    jobs = [
        (
            args.dataset,
            str(args.root),
            {
                "dataset": r.dataset,
                "video_id": r.video_id,
                "subject": r.subject,
                "paths": {k: str(v) for k, v in r.paths.items()},
                "meta": r.meta,
            },
            str(out_dir),
            args.overwrite,
        )
        for r in recs
    ]
    print(f"{len(jobs)} videos -> {out_dir} with {args.workers} workers")
    log_path = out_dir / "extract_log.jsonl"
    with mp.get_context("spawn").Pool(args.workers) as pool, log_path.open("a") as log:
        for res in pool.imap_unordered(_process, jobs):
            log.write(json.dumps(res) + "\n")
            log.flush()
            flag = (
                ""
                if res["status"] != "ok" or res.get("align_corr", 1) >= 0.3
                else "  <-- low GT alignment, inspect"
            )
            print({k: v for k, v in res.items() if k != "trace"}, flag)


if __name__ == "__main__":
    main()
