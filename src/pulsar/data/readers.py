"""Dataset readers for UBFC-rPPG (Dataset 2), PURE and MMPD.

Layouts follow the datasets' documentation and rPPG-Toolbox's loaders. Details marked VERIFY
must be confirmed on first download with `scripts/inspect_dataset.py`; readers fail loudly with
an explanatory error rather than guessing.

Each reader yields `VideoRecord`s; `iter_frames(rec)` yields (timestamp_s, frame_rgb_uint8) and
`load_gt(rec)` returns a `GroundTruth` on the same clock as the frame timestamps.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VideoRecord:
    dataset: str
    video_id: str
    subject: str
    paths: dict[str, Path]
    meta: dict = field(default_factory=dict)


@dataclass
class GroundTruth:
    t: np.ndarray  # [N] seconds
    ppg: np.ndarray  # [N]
    hr: np.ndarray | None = None
    spo2: np.ndarray | None = None


class DatasetFormatError(RuntimeError):
    pass


def _cv2():
    import cv2

    return cv2


# ------------------------------------------------------------------------ UBFC-rPPG
class UBFCReader:
    """root/subjectN/{vid.avi, ground_truth.txt}. ground_truth.txt rows: PPG, HR, time (s)."""

    name = "ubfc"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_videos(self) -> list[VideoRecord]:
        recs = []
        for d in sorted(self.root.glob("subject*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            vid, gt = d / "vid.avi", d / "ground_truth.txt"
            if vid.exists() and gt.exists():
                recs.append(VideoRecord(self.name, d.name, d.name, {"video": vid, "gt": gt}))
        if not recs:
            raise DatasetFormatError(f"no UBFC subject*/vid.avi + ground_truth.txt under {self.root}")
        return recs

    def load_gt(self, rec: VideoRecord) -> GroundTruth:
        rows = [
            np.array(r.split(), dtype=np.float64)
            for r in rec.paths["gt"].read_text().splitlines()
            if r.strip()
        ]
        if len(rows) < 3:
            raise DatasetFormatError(f"{rec.paths['gt']}: expected 3 rows (PPG, HR, time)")
        ppg, hr, t = rows[0], rows[1], rows[2]
        t = t - t[0]
        return GroundTruth(t=t, ppg=ppg, hr=hr)

    def iter_frames(self, rec: VideoRecord) -> Iterator[tuple[float, np.ndarray]]:
        cv2 = _cv2()
        cap = cv2.VideoCapture(str(rec.paths["video"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # VERIFY: UBFC GT is sampled per frame, so its time vector doubles as frame timestamps.
        gt_t = self.load_gt(rec).t
        i = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                t = float(gt_t[i]) if i < len(gt_t) else i / fps
                yield t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                i += 1
        finally:
            cap.release()


# ------------------------------------------------------------------------ PURE
class PUREReader:
    """root/XX-YY/XX-YY.json + root/XX-YY/XX-YY/Image<ns>.png (VERIFY nesting on download).

    JSON: {"/FullPackage": [{"Timestamp": ns, "Value": {"waveform", "pulseRate", "o2saturation",
    ...}}], "/Image": [{"Timestamp": ns}, ...]}.
    """

    name = "pure"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def list_videos(self) -> list[VideoRecord]:
        recs = []
        for d in sorted(self.root.glob("??-??")):
            js = d / f"{d.name}.json"
            img_dir = d / d.name if (d / d.name).is_dir() else d
            if js.exists() and any(img_dir.glob("Image*.png")):
                subj, task = d.name.split("-")
                recs.append(
                    VideoRecord(
                        self.name,
                        d.name,
                        subj,
                        {"json": js, "frames": img_dir},
                        meta={"task": int(task), "activity": PURE_TASKS.get(int(task), "unknown")},
                    )
                )
        if not recs:
            raise DatasetFormatError(f"no PURE XX-YY/XX-YY.json + Image*.png under {self.root}")
        return recs

    @staticmethod
    def _frame_files(rec: VideoRecord) -> list[tuple[int, Path]]:
        out = []
        for p in rec.paths["frames"].glob("Image*.png"):
            m = re.match(r"Image(\d+)\.png$", p.name)
            if m:
                out.append((int(m.group(1)), p))
        return sorted(out)

    def _t0(self, rec: VideoRecord) -> int:
        return self._frame_files(rec)[0][0]

    def load_gt(self, rec: VideoRecord) -> GroundTruth:
        data = json.loads(rec.paths["json"].read_text())
        pk = data.get("/FullPackage")
        if pk is None:
            raise DatasetFormatError(f"{rec.paths['json']}: missing '/FullPackage'")
        t0 = self._t0(rec)
        ts = np.array([e["Timestamp"] for e in pk], dtype=np.float64)
        v = [e["Value"] for e in pk]
        return GroundTruth(
            t=(ts - t0) / 1e9,
            ppg=np.array([x["waveform"] for x in v], dtype=np.float64),
            hr=np.array([x.get("pulseRate", np.nan) for x in v], dtype=np.float64),
            spo2=np.array([x.get("o2saturation", np.nan) for x in v], dtype=np.float64),
        )

    def iter_frames(self, rec: VideoRecord) -> Iterator[tuple[float, np.ndarray]]:
        cv2 = _cv2()
        files = self._frame_files(rec)
        t0 = files[0][0]
        for ts, p in files:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            yield (ts - t0) / 1e9, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


PURE_TASKS = {
    1: "steady",
    2: "talking",
    3: "slow_translation",
    4: "fast_translation",
    5: "small_rotation",
    6: "medium_rotation",
}


# ------------------------------------------------------------------------ MMPD
class MMPDReader:
    """root/subjectN/p{N}_{k}.mat with keys video [T,H,W,3] float, GT_ppg, light, motion,
    exercise, skin_color, gender, glasser, hair_cover, makeup (VERIFY on download).
    """

    name = "mmpd"
    fps = 30.0  # VERIFY
    META_KEYS = ("light", "motion", "exercise", "skin_color", "gender", "glasser", "hair_cover", "makeup")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: tuple[Path, dict] | None = None  # .mat files are large; load once per video

    def list_videos(self) -> list[VideoRecord]:
        recs = []
        for d in sorted(self.root.glob("subject*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            for m in sorted(d.glob("*.mat")):
                recs.append(VideoRecord(self.name, m.stem, d.name, {"mat": m}))
        if not recs:
            raise DatasetFormatError(f"no MMPD subject*/*.mat under {self.root}")
        return recs

    def _load(self, rec: VideoRecord) -> dict:
        from scipy.io import loadmat

        path = rec.paths["mat"]
        if self._cache is not None and self._cache[0] == path:
            return self._cache[1]
        try:
            mat = loadmat(str(path))
        except NotImplementedError as e:  # MATLAB v7.3 (HDF5)
            raise DatasetFormatError(f"{path} is MATLAB v7.3; add an h5py path") from e
        self._cache = (path, mat)
        return mat

    @staticmethod
    def _scalar(v):
        a = np.asarray(v).squeeze()
        if a.dtype.kind in "US":
            return str(a)
        if a.dtype == object:
            return str(a.item()) if a.size == 1 else [str(x) for x in a.ravel()]
        return a.item() if a.size == 1 else a.tolist()

    def read_meta(self, rec: VideoRecord, mat: dict | None = None) -> dict:
        mat = mat if mat is not None else self._load(rec)
        meta = {k: self._scalar(mat[k]) for k in self.META_KEYS if k in mat}
        if "skin_color" in meta:
            meta["fitzpatrick"] = int(meta["skin_color"])
        return meta

    def load_gt(self, rec: VideoRecord) -> GroundTruth:
        mat = self._load(rec)
        ppg = np.asarray(mat["GT_ppg"], dtype=np.float64).reshape(-1)
        n_frames = np.asarray(mat["video"]).shape[0]
        # VERIFY: GT_ppg assumed to span the video duration (toolbox resamples it to frame count).
        t = np.linspace(0, (n_frames - 1) / self.fps, ppg.size)
        return GroundTruth(t=t, ppg=ppg)

    def iter_frames(self, rec: VideoRecord) -> Iterator[tuple[float, np.ndarray]]:
        video = np.asarray(self._load(rec)["video"])
        scale = 255.0 if video.dtype.kind == "f" and video.max() <= 1.0 + 1e-6 else 1.0
        for i in range(video.shape[0]):
            f = video[i]
            yield i / self.fps, np.clip(np.round(f * scale), 0, 255).astype(np.uint8)


READERS = {"ubfc": UBFCReader, "pure": PUREReader, "mmpd": MMPDReader}


def get_reader(name: str, root: str | Path):
    try:
        return READERS[name](root)
    except KeyError as e:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(READERS)}") from e
