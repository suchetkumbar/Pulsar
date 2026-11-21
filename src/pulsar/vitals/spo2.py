"""Experimental SpO2 from RGB face traces: ratio-of-ratios with a matched-filter AC estimate (docs 04 §5).

    RR = (AC_R / DC_R) / (AC_B / DC_B)            SpO2 ~= A - B * RR   (A, B fitted per training fold)

AC is the projection of each band-passed channel onto a reference BVP (POS of the face mean, or
TraceNet's predicted BVP), which rejects noise that is not phase-locked to the pulse. DC is the
channel mean. RR is computed per patch and pooled with a weighted median in log space; weights are
(fraction of green-channel band energy explained by the reference) x (skin pixel count).

Honesty constraints (docs 04 §5, 06): RGB cameras lack the red/IR pair oximeters rely on; public
data (PURE) spans only ~93-99 %, so a constant-mean predictor already scores a low MAE and every
result here is reported against it; hypoxaemic ranges are untested; the reference oximeter itself
over-reads on darker skin (Sjoding et al., NEJM 2020). This is a research probe, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pulsar.data.windows import Windows
from pulsar.eval.metrics import hr_metrics, subject_level
from pulsar.signal.classical import pos
from pulsar.signal.filters import HR_BAND_HZ, bandpass

CHANNELS = {"r": 0, "g": 1, "b": 2}
PAIRS = (("r", "b"), ("r", "g"))


def reference_bvp(w: Windows, fs: float = 30.0, band: tuple[float, float] = HR_BAND_HZ) -> np.ndarray:
    """POS on the pixel-weighted face-mean trace of every window -> [N, T] band-passed reference."""
    rgb = np.nan_to_num(w.face_mean_rgb(), nan=0.0)
    if len(w) == 0:
        return np.zeros((0, w.x.shape[1]))
    return np.stack([bandpass(pos(r, fs), fs, band) for r in rgb])


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def window_rr(
    x: np.ndarray,
    valid: np.ndarray,
    n_pix: np.ndarray,
    ref: np.ndarray,
    fs: float = 30.0,
    band: tuple[float, float] = HR_BAND_HZ,
    pair: tuple[str, str] = ("r", "b"),
    min_valid_frac: float = 0.9,
) -> dict[str, float | np.ndarray]:
    """Ratio-of-ratios for one window.

    x [T, P, 3] linear RGB, valid [T, P], n_pix [P], ref [T]. Returns pooled ``rr`` and ``quality``
    plus per-patch ``rr_patch`` / ``q_patch`` (NaN where a patch was rejected).
    """
    p_n = x.shape[1]
    nan_out = {"rr": np.nan, "quality": np.nan, "rr_patch": np.full(p_n, np.nan), "q_patch": np.zeros(p_n)}
    y = np.asarray(ref, dtype=np.float64)
    y = y - y.mean()
    nrm = np.linalg.norm(y)
    if not np.isfinite(nrm) or nrm < 1e-12:
        return nan_out
    y = y / nrm

    v = valid.astype(bool)
    keep = v.mean(axis=0) >= min_valid_frac  # [P]
    cnt = np.maximum(v.sum(axis=0), 1)[:, None]  # [P, 1]
    xd = np.asarray(x, dtype=np.float64)
    dc = (xd * v[..., None]).sum(axis=0) / cnt  # [P, 3]
    filled = np.where(v[..., None], xd, dc[None])  # invalid samples -> patch DC (no spurious AC)
    xf = bandpass(filled, fs, band, axis=0)  # [T, P, 3]
    proj = np.einsum("tpc,t->pc", xf, y)  # signed matched-filter amplitude
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = proj / dc
        i, j = CHANNELS[pair[0]], CHANNELS[pair[1]]
        rr_p = np.abs(rel[:, i]) / np.abs(rel[:, j])
        e_g = (xf[:, :, 1] ** 2).sum(axis=0)
        q_p = np.where(e_g > 0, proj[:, 1] ** 2 / e_g, 0.0)
    same_sign = np.sign(rel[:, i]) == np.sign(rel[:, j])
    ok = keep & same_sign & np.isfinite(rr_p) & (rr_p > 0) & (dc[:, i] > 0) & (dc[:, j] > 0)
    wts = q_p * np.asarray(n_pix, dtype=np.float64) * ok
    rr_patch = np.where(ok, rr_p, np.nan)
    if not ok.any() or wts.sum() <= 0:
        return nan_out | {"rr_patch": rr_patch, "q_patch": np.where(ok, q_p, 0.0)}
    rr = float(np.exp(_weighted_median(np.log(rr_p[ok]), wts[ok])))
    quality = float(np.average(q_p[ok], weights=np.asarray(n_pix, float)[ok]))
    return {"rr": rr, "quality": quality, "rr_patch": rr_patch, "q_patch": np.where(ok, q_p, 0.0)}


def window_features(
    ws: list[Windows],
    fs: float = 30.0,
    refs: list[np.ndarray] | None = None,
    pairs: tuple[tuple[str, str], ...] = PAIRS,
    ref_name: str = "pos",
) -> pd.DataFrame:
    """One row per window: identifiers, SpO2 target and RR features (``rr_rb``, ``rr_rg``, ``q``)."""
    rows = []
    for k, w in enumerate(ws):
        if len(w) == 0:
            continue
        ref = reference_bvp(w, fs) if refs is None else refs[k]
        sat = w.spo2 if w.spo2 is not None else np.full(len(w), np.nan)
        for n in range(len(w)):
            row = {
                "dataset": w.meta.get("dataset"),
                "subject": w.subject,
                "video_id": w.video_id,
                "start_s": float(w.start_s[n]),
                "ita": w.meta.get("ita"),
                "fitzpatrick": w.meta.get("fitzpatrick"),
                "spo2_true": float(sat[n]),
                "ref": ref_name,
            }
            for a, b in pairs:
                r = window_rr(w.x[n], w.valid[n], w.n_pix[n], ref[n], fs, pair=(a, b))
                row[f"rr_{a}{b}"] = r["rr"]
                row["q"] = r["quality"]
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class LinearCalibration:
    """SpO2 = a + b * RR (b is expected negative). Least squares, optionally quality-weighted."""

    a: float = float("nan")
    b: float = 0.0

    def fit(self, rr: np.ndarray, spo2: np.ndarray, weights: np.ndarray | None = None) -> LinearCalibration:
        rr, spo2 = np.asarray(rr, float), np.asarray(spo2, float)
        ok = np.isfinite(rr) & np.isfinite(spo2)
        w = None if weights is None else np.sqrt(np.clip(np.asarray(weights, float)[ok], 0, None))
        if ok.sum() < 3 or np.ptp(rr[ok]) == 0:
            self.a, self.b = float(np.mean(spo2[ok])) if ok.any() else float("nan"), 0.0
            return self
        self.b, self.a = (float(c) for c in np.polyfit(rr[ok], spo2[ok], 1, w=w))
        return self

    def predict(self, rr: np.ndarray) -> np.ndarray:
        return np.clip(self.a + self.b * np.asarray(rr, float), 70.0, 100.0)


def loso_predict(
    df: pd.DataFrame, feature: str | None, target: str = "spo2_true", group: str = "subject"
) -> np.ndarray:
    """Leave-one-subject-out predictions. feature=None -> constant training-fold mean (the baseline)."""
    pred = np.full(len(df), np.nan)
    y = df[target].to_numpy(float)
    g = df[group].to_numpy()
    for s in pd.unique(g):
        te = g == s
        tr = ~te & np.isfinite(y)
        if feature is None:
            pred[te] = np.nanmean(y[tr])
            continue
        x = df[feature].to_numpy(float)
        tr &= np.isfinite(x)
        cal = LinearCalibration().fit(x[tr], y[tr])
        pred[te] = cal.predict(x[te])
    return pred


def spo2_metrics(df: pd.DataFrame, pred: str, true: str = "spo2_true") -> dict[str, float]:
    """Window-pooled MAE/RMSE/Pearson plus subject-level MAE (subject = independent unit)."""
    d = df[np.isfinite(df[pred]) & np.isfinite(df[true])]
    m = hr_metrics(d[pred].to_numpy(), d[true].to_numpy())
    subj = subject_level(d, pred=pred, true=true)
    return {
        "subject_mae": float(subj["mae"].iloc[0]) if len(d) else float("nan"),
        "window_mae": m["mae"],
        "rmse": m["rmse"],
        "pearson": m["pearson"],
        "n_windows": int(len(d)),
        "n_subjects": int(d["subject"].nunique()),
    }


def evaluate_loso(df: pd.DataFrame, features: tuple[str, ...] = ("rr_rb", "rr_rg")) -> pd.DataFrame:
    """Constant-mean baseline vs each RR feature under leave-one-subject-out calibration."""
    d = df[np.isfinite(df["spo2_true"])].reset_index(drop=True)
    rows = []
    for name, feat in [("constant-mean", None)] + [(f"RoR {f[3:].upper()}", f) for f in features]:
        col = f"pred_{feat or 'const'}"
        d[col] = loso_predict(d, feat)
        rows.append({"method": name} | spo2_metrics(d, col))
    return pd.DataFrame(rows).set_index("method")
