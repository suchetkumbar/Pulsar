"""The Android core's committed parity fixtures must match the current Python preprocessing."""

import gzip
import json
from pathlib import Path

import numpy as np

from pulsar.export.fixtures import extractor_case, window_case, write_fixtures

COMMITTED = Path(__file__).resolve().parents[1] / "android/core/src/test/resources/fixtures"
REGEN = "uv run --no-sync python scripts/make_android_fixtures.py"


def _load(path: Path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_fixtures_are_current(tmp_path):
    fresh = write_fixtures(tmp_path)
    for p in fresh:
        committed = COMMITTED / p.name
        assert committed.is_file(), f"missing {committed}; run: {REGEN}"
        assert _load(committed) == _load(p), f"{p.name} is stale (Python preprocessing changed); run: {REGEN}"


def test_extractor_case_covers_the_interesting_paths():
    case = extractor_case()
    frames = [f["expected"] for f in case["frames"]]
    assert sum(f["face_found"] for f in frames) == 5  # 5 detections, 4 carried, 1 empty
    assert sum(1 for f in frames if not f["face_found"] and any(f["valid"])) == 4
    assert not any(frames[8]["valid"]) and frames[8]["boxes"] is None
    yaw = frames[3]
    assert max(yaw["view_angles"]) > case["patch_spec"]["max_view_angle_deg"]  # some patches rejected
    assert sum(yaw["valid"]) < sum(frames[0]["valid"])
    # masks matter: some patches lose pixels to hair / glints / the non-skin blob
    box_px = [(b[2] - b[0]) * (b[3] - b[1]) for b in frames[0]["boxes"]]
    assert sum(n < a for n, a in zip(frames[0]["n_pix"], box_px, strict=True)) >= 6
    assert frames[0]["motion"] == 0.0 and frames[1]["motion"] > 0.0


def test_window_case_is_self_consistent():
    case, (tg, x, v, npx, mot) = window_case("medium_gaps", "medium", 82.0, 3.0, 0.0, (5,), seed=102)
    e = case["expected"]
    assert np.allclose(np.asarray(e["x"]), x, atol=1e-9)
    assert not np.asarray(e["valid"])[:, 5].any()  # invalid patch stays invalid
    assert len(e["t_grid"]) == case["window"]["t_len"] == 300
