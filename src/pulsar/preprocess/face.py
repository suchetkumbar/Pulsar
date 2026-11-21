"""MediaPipe Face Landmarker wrapper (Tasks API, same model family the Android app uses).

The model file is downloaded once to data/assets/ (gitignored). A `Landmarker` protocol lets the
trace extractor be tested with injected synthetic landmarks.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("data/assets/face_landmarker.task")


class Landmarker(Protocol):
    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Return normalised landmarks [478, 3] (x, y in [0,1], z ~ x-scale) or None."""


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)  # public model asset, no user data sent
        tmp.replace(path)
    return path


class MediaPipeLandmarker:
    """Face Landmarker in VIDEO mode (monotonic timestamps, temporal tracking)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(Path(model_path)))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        ts = max(int(timestamp_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing ts
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self._lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._lm.close()

    def __enter__(self) -> MediaPipeLandmarker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
