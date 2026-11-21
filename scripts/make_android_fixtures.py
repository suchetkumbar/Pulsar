"""Regenerate the Android core's parity fixtures (synthetic only; safe to commit).

    uv run --no-sync python scripts/make_android_fixtures.py

Writes android/core/src/test/resources/fixtures/{extractor,window_*}.json.gz. Re-run after any change to
pulsar.preprocess (patches, skin mask, traces, resample, live) and commit the result with the change, so
the Kotlin tests keep checking the port against the current Python reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pulsar.export.fixtures import write_fixtures

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "android/core/src/test/resources/fixtures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    for p in write_fixtures(args.out):
        print(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
