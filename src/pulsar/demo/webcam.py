"""Live webcam heart-rate demo (desktop, Windows native). NOT A MEDICAL DEVICE.

    uv run python -m pulsar.demo.webcam                       # classical POS (no model needed)
    uv run python -m pulsar.demo.webcam --model runs/<run>/best.pt
    uv run python -m pulsar.demo.webcam --video clip.mp4      # offline file instead of a camera

Keys: q = quit, s = save the session log (time, HR, confidence) to outputs/demo/ for comparison
with a pulse oximeter.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from pulsar.preprocess.face import MediaPipeLandmarker
from pulsar.preprocess.live import live_window
from pulsar.preprocess.patches import PatchSpec
from pulsar.preprocess.traces import TraceExtractor
from pulsar.signal.classical import pos
from pulsar.signal.filters import postprocess_bvp
from pulsar.signal.spectral import hr_grid, snr_db

FS = 30.0
WIN_S = 10.0
DISCLAIMER = "Research prototype - not a medical device"


class Estimator:
    """Turns the last WIN_S seconds of patch traces into (hr, confidence, bvp, patch_weights)."""

    def __init__(self, model_path: str | None, spec: PatchSpec):
        self.spec = spec
        self.model = None
        if model_path:
            import torch

            from pulsar.train.loop import load_model

            self.torch = torch
            self.model = load_model(model_path, "cpu")

    def __call__(self, t, rgb, valid, n_pix, motion):
        win = live_window(t, rgb, valid, n_pix, motion, FS, WIN_S)
        if win is None:
            return None
        _, x, v, npx, motion = win
        if self.model is not None:
            return self._model(x, v, npx, motion)
        return self._pos(x, v, npx)

    def _pos(self, x, v, npx):
        keep = v.mean(0) >= 0.8
        if keep.sum() < 4:
            return None
        w = v * npx[None] * keep[None]
        rgb = (w[..., None] * x).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        bvp = postprocess_bvp(pos(rgb, FS), FS)
        hr = hr_grid(bvp, FS)
        snr = snr_db(bvp, hr, FS)
        conf = float(1 / (1 + np.exp(-(snr - 0.0) / 2.0)))  # heuristic SNR->[0,1] until C4 calibration
        per_patch = np.zeros(x.shape[1])
        for k in np.nonzero(keep)[0]:
            per_patch[k] = snr_db(postprocess_bvp(pos(x[:, k], FS), FS), hr, FS)
        weights = np.exp(per_patch / 3.0) * keep
        return hr, conf, bvp, weights / max(weights.sum(), 1e-9)

    def _model(self, x, v, npx, motion):
        from pulsar.models.tracenet import hr_from_logits

        th = self.torch
        with th.no_grad():
            out = self.model(
                th.from_numpy(x.transpose(1, 0, 2)[None]).float(),
                th.from_numpy(v.T[None].copy()),
                th.from_numpy(npx[None]).float(),
                th.tensor([motion], dtype=th.float32),
            )
            hr, _, conf = hr_from_logits(out["hr_logits"], self.model.bpm, self.model.temperature)
        return float(hr[0]), float(conf[0]), out["bvp"][0].numpy(), out["alpha"][0].numpy()


def draw(cv2, frame, boxes, weights, hr, conf, bvp, abstain_msg):
    h, w = frame.shape[:2]
    if boxes is not None:
        wmax = max(float(np.max(weights)), 1e-9) if weights is not None else 1.0
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            a = 0.0 if weights is None else float(weights[k]) / wmax
            color = (int(80 * (1 - a)), int(80 + 175 * a), int(255 * (1 - a)))  # BGR: red -> green
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
    panel_h = 90
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    if bvp is not None:
        b = (bvp - bvp.min()) / max(np.ptp(bvp), 1e-9)
        xs = np.linspace(10, w * 0.55, len(b)).astype(int)
        ys = (h - 15 - b * (panel_h - 30)).astype(int)
        cv2.polylines(frame, [np.stack([xs, ys], 1)], False, (90, 220, 90), 1)
    text = abstain_msg or (f"{hr:5.1f} bpm" if hr is not None else "warming up...")
    cv2.putText(frame, text, (int(w * 0.6), h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if conf is not None and not abstain_msg:
        cv2.putText(
            frame,
            f"confidence {conf:.2f}",
            (int(w * 0.6), h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
    cv2.putText(frame, DISCLAIMER, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


def main() -> None:
    import cv2

    from pulsar.preprocess.patches import landmarks_to_pixels, patch_boxes

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--min-conf", type=float, default=0.4)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    spec = PatchSpec()
    ex = TraceExtractor(spec)
    est = Estimator(args.model, spec)
    buf: collections.deque = collections.deque()
    log_rows: list[tuple[float, float, float]] = []
    t_start = time.perf_counter()
    last_est = 0.0
    hr = conf = bvp = weights = None
    fps_hist: collections.deque = collections.deque(maxlen=60)
    with MediaPipeLandmarker() as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if args.video else time.perf_counter() - t_start
            fps_hist.append(t)
            rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lmk = lm.detect(rgb_frame, int(t * 1000))
            st = ex.process(rgb_frame, lmk)
            buf.append((t, st.rgb_lin, st.valid, st.n_pix, st.motion))
            while buf and buf[0][0] < t - (WIN_S + 2):
                buf.popleft()
            boxes = None
            if lmk is not None:
                h, w = rgb_frame.shape[:2]
                boxes = patch_boxes(landmarks_to_pixels(lmk, w, h), spec, w, h)
            abstain = None if st.face_found else "No face - look at the camera"
            if t - last_est >= 1.0 and len(buf) > FS * WIN_S * 0.8:
                last_est = t
                ts = np.array([b[0] for b in buf])
                res = est(
                    ts,
                    np.stack([b[1] for b in buf]),
                    np.stack([b[2] for b in buf]),
                    np.stack([b[3] for b in buf]).astype(float),
                    np.array([b[4] for b in buf], dtype=float),
                )
                if res is not None:
                    hr, conf, bvp, weights = res
                    log_rows.append((t, hr, conf))
            if abstain is None and conf is not None and conf < args.min_conf:
                abstain = "Low signal - hold still / add light"
            if len(fps_hist) > 1:
                fps = (len(fps_hist) - 1) / max(fps_hist[-1] - fps_hist[0], 1e-6)
                cv2.putText(
                    bgr,
                    f"{fps:4.1f} fps",
                    (bgr.shape[1] - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
            cv2.imshow("Pulsar", draw(cv2, bgr, boxes, weights, hr, conf, bvp, abstain))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and log_rows:
                out = Path("outputs/demo") / time.strftime("session_%Y%m%d-%H%M%S.csv")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    "t_s,hr_bpm,confidence\n" + "\n".join(f"{a:.2f},{b:.1f},{c:.3f}" for a, b, c in log_rows)
                )
                print(f"saved {out}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
