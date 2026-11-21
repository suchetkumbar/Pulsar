"""Training-free rPPG algorithms operating on mean-RGB traces of shape [T, 3].

Implementations follow the original papers and rPPG-Toolbox's reference code:
* GREEN  - Verkruysse et al., Optics Express 2008
* ICA    - Poh et al., Optics Express 2010 (FastICA here; toolbox uses JADE)
* CHROM  - de Haan & Jeanne, IEEE TBME 2013 (windowed, Hann overlap-add)
* POS    - Wang et al., IEEE TBME 2017 (1.6 s windows, overlap-add)
* LGI    - Pilz et al., CVPRW 2018 (local group invariance projection)

All return a raw BVP estimate of length T; apply `filters.postprocess_bvp` before HR estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal as sps

_POS_P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
_EPS = 1e-9


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"expected [T, 3] RGB trace, got {rgb.shape}")
    return rgb


def green(rgb: np.ndarray, fs: float | None = None) -> np.ndarray:
    rgb = _check_rgb(rgb)
    return rgb[:, 1] - rgb[:, 1].mean()


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (Wang et al. 2017), vectorised overlap-add."""
    rgb = _check_rgb(rgb)
    t_len = rgb.shape[0]
    l = int(np.ceil(win_s * fs))
    out = np.zeros(t_len)
    if t_len < l:
        return out
    win = sliding_window_view(rgb, l, axis=0)  # [N, 3, l]
    cn = win / (win.mean(axis=2, keepdims=True) + _EPS)
    s = np.einsum("ij,njl->nil", _POS_P, cn)  # [N, 2, l]
    std0 = s[:, 0].std(axis=1)
    std1 = s[:, 1].std(axis=1)
    alpha = std0 / (std1 + _EPS)
    h = s[:, 0] + alpha[:, None] * s[:, 1]
    h = h - h.mean(axis=1, keepdims=True)
    for m in range(h.shape[0]):  # overlap-add (N small loops over length-l rows is cheap)
        out[m : m + l] += h[m]
    return out


def chrom(rgb: np.ndarray, fs: float, win_s: float = 1.6, band=(0.7, 2.5)) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013) with per-window band-pass and Hann overlap-add."""
    rgb = _check_rgb(rgb)
    t_len = rgb.shape[0]
    wl = int(np.ceil(win_s * fs))
    wl += wl % 2
    half = wl // 2
    nyq = fs / 2
    b, a = sps.butter(3, [band[0] / nyq, band[1] / nyq], btype="bandpass")
    hann = np.hanning(wl)
    out = np.zeros(t_len)
    start = 0
    while start + wl <= t_len:
        seg = rgb[start : start + wl]
        cn = seg / (seg.mean(axis=0) + _EPS) - 1.0
        xs = 3 * cn[:, 0] - 2 * cn[:, 1]
        ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf = sps.filtfilt(b, a, xs)
        yf = sps.filtfilt(b, a, ys)
        alpha = xf.std() / (yf.std() + _EPS)
        out[start : start + wl] += (xf - alpha * yf) * hann
        start += half
    return out


def lgi(rgb: np.ndarray, fs: float | None = None) -> np.ndarray:
    """Local Group Invariance (Pilz et al. 2018): project out the dominant colour direction."""
    rgb = _check_rgb(rgb)
    c = (rgb / (rgb.mean(axis=0) + _EPS)).T  # [3, T]
    u, _, _ = np.linalg.svd(c, full_matrices=False)
    s = u[:, :1]
    p = np.eye(3) - s @ s.T
    y = p @ c
    return y[1] - y[1].mean()


def ica(rgb: np.ndarray, fs: float, band=(0.7, 2.5), seed: int = 0) -> np.ndarray:
    """ICA (Poh et al. 2010): pick the independent component with the strongest in-band peak."""
    from sklearn.decomposition import FastICA

    from pulsar.signal.filters import detrend_tarvainen

    rgb = _check_rgb(rgb)
    x = np.stack([detrend_tarvainen(rgb[:, k], 100.0) for k in range(3)], axis=1)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + _EPS)
    comps = FastICA(n_components=3, whiten="unit-variance", random_state=seed, max_iter=1000)
    s = comps.fit_transform(x)  # [T, 3]
    f, pxx = sps.periodogram(s, fs=fs, axis=0)
    m = (f >= band[0]) & (f <= band[1])
    k = int(np.argmax(pxx[m].max(axis=0)))
    return s[:, k]


METHODS = {"GREEN": green, "POS": pos, "CHROM": chrom, "LGI": lgi, "ICA": ica}


def run_method(name: str, rgb: np.ndarray, fs: float) -> np.ndarray:
    try:
        fn = METHODS[name.upper()]
    except KeyError as e:
        raise ValueError(f"unknown method {name!r}; choose from {sorted(METHODS)}") from e
    return fn(rgb, fs)
