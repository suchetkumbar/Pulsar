"""Minimal experiment tracker: one directory per run with config, metrics CSV and artifacts.

Replaces MLflow (too heavy for the current network): runs/<UTC time>_<name>/
    config.json   resolved config + environment
    metrics.csv   one row per epoch
    summary.json  final/best metrics
    *.pt          checkpoints
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from pathlib import Path


def _git_rev() -> str | None:
    try:  # read-only; git operations are the user's
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


class RunLog:
    def __init__(self, name: str, config: dict, root: str | Path = "runs"):
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.dir = Path(root) / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        env = {"python": platform.python_version(), "git_rev": _git_rev(), "host_os": platform.system()}
        (self.dir / "config.json").write_text(
            json.dumps({"config": config, "env": env}, indent=2, default=str)
        )
        self._fields: list[str] | None = None

    def log(self, row: dict) -> None:
        path = self.dir / "metrics.csv"
        if self._fields is None:
            self._fields = list(row)
            with path.open("w", newline="") as f:
                csv.DictWriter(f, self._fields).writeheader()
        with path.open("a", newline="") as f:
            csv.DictWriter(f, self._fields, extrasaction="ignore").writerow(row)

    def summary(self, d: dict) -> None:
        (self.dir / "summary.json").write_text(json.dumps(d, indent=2, default=str))
