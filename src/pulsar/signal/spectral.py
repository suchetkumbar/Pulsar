"""Heart-rate and SNR estimation from BVP signals.

Two families:
* `*_toolbox` functions replicate rPPG-Toolbox exactly (periodogram, nfft = next power of 2,
  0.75-2.5 Hz band, its SNR definition) -- use for baseline parity / published comparability.
* `hr_grid` / `snr_db` evaluate the spectrum on an exact 1-bpm grid (the same grid as TraceNet's
  DFT head), which avoids the ~3.5 bpm bin width of a 300-sample FFT. Used for Pulsar metrics.
  Ground truth and predictions must always go through the *same* function.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from pulsar.signal.filters import HR_BAND_HZ, TOOLBOX_BAND_HZ


def next_pow2(n: int) -> int:
    return 1 if n == 0 else 2 ** int(np.ceil(np.log2(n)))


# ---------------------------------------------------------------- toolbox-compatible
def hr_fft_toolbox(x: np.ndarray, fs: float, band: tuple[float, float] = TOOLBOX_BAND_HZ) -> float:
    """HR (bpm) = periodogram peak in band, nfft = next power of two (rPPG-Toolbox)."""
    x = np.asarray(x, dtype=np.float64)
    f, pxx = sps.periodogram(x, fs=fs, nfft=next_pow2(len(x)), detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    return float(f[m][np.argmax(pxx[m])] * 60.0)


def snr_db_toolbox(
    x: np.ndarray, hr_bpm: float, fs: float, band: tuple[float, float] = TOOLBOX_BAND_HZ
) -> float:
    """rPPG-Toolbox `_calculate_SNR`: (sum pxx^2 at f0,2f0 +-6 bpm) / (sum pxx^2 elsewhere), 20log10."""
    f0 = hr_bpm / 60.0
    dev = 6.0 / 60.0
    x = np.asarray(x, dtype=np.float64)
    f, pxx = sps.periodogram(x, fs=fs, nfft=next_pow2(len(x)), detrend=False)
    h1 = (f >= f0 - dev) & (f <= f0 + dev)
    h2 = (f >= 2 * f0 - dev) & (f <= 2 * f0 + dev)
    rem = (f >= band[0]) & (f <= band[1]) & ~h1 & ~h2
    sig = np.sum(pxx[h1] ** 2) + np.sum(pxx[h2] ** 2)
    noise = np.sum(pxx[rem] ** 2)
    if noise == 0 or sig == 0:
        return float("nan")
    return float(20.0 * np.log10(sig / noise))


# ---------------------------------------------------------------- exact-grid (Pulsar)
def bpm_grid(lo_bpm: float = 40.0, hi_bpm: float = 200.0, step: float = 1.0) -> np.ndarray:
    return np.arange(lo_bpm, hi_bpm + step / 2, step, dtype=np.float64)


def dft_power(x: np.ndarray, fs: float, freqs_hz: np.ndarray, window: str = "hann") -> np.ndarray:
    """Power of x at arbitrary frequencies (direct DFT). x: [..., T] -> [..., K]."""
    x = np.asarray(x, dtype=np.float64)
    t_len = x.shape[-1]
    w = sps.get_window(window, t_len, fftbins=False) if window else np.ones(t_len)
    xw = (x - x.mean(axis=-1, keepdims=True)) * w
    t = np.arange(t_len) / fs
    phase = 2j * np.pi * np.outer(t, freqs_hz)  # [T, K]
    spec = xw @ np.exp(-phase)
    return np.abs(spec) ** 2


def hr_grid(x: np.ndarray, fs: float, grid_bpm: np.ndarray | None = None, refine: bool = True) -> float:
    """HR (bpm) = peak of the DFT power on an exact bpm grid (+ parabolic refinement)."""
    grid = bpm_grid() if grid_bpm is None else np.asarray(grid_bpm, dtype=np.float64)
    p = dft_power(x, fs, grid / 60.0)
    k = int(np.argmax(p))
    if refine and 0 < k < len(grid) - 1:
        a, b, c = np.log(p[k - 1 : k + 2] + 1e-30)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        step = grid[1] - grid[0]
        return float(grid[k] + np.clip(delta, -0.5, 0.5) * step)
    return float(grid[k])


def snr_db(
    x: np.ndarray,
    hr_bpm: float,
    fs: float,
    band: tuple[float, float] = HR_BAND_HZ,
    dev_hz: float = 0.1,
    resolution_hz: float = 0.01,
) -> float:
    """de Haan & Jeanne (2013)-style SNR: energy within +-dev of f0 and 2f0 vs rest of band (dB)."""
    freqs = np.arange(band[0], band[1] + resolution_hz / 2, resolution_hz)
    p = dft_power(x, fs, freqs)
    f0 = hr_bpm / 60.0
    mask = (np.abs(freqs - f0) <= dev_hz) | (np.abs(freqs - 2 * f0) <= dev_hz)
    sig, noise = p[mask].sum(), p[~mask].sum()
    if noise <= 0 or sig <= 0:
        return float("nan")
    return float(10.0 * np.log10(sig / noise))


def hr_welch(x: np.ndarray, fs: float, band: tuple[float, float] = HR_BAND_HZ) -> float:
    """HR via Welch PSD (robust for long ground-truth segments)."""
    x = np.asarray(x, dtype=np.float64)
    nper = min(len(x), int(8 * fs))
    f, pxx = sps.welch(x, fs=fs, nperseg=nper, nfft=max(4096, next_pow2(nper)), detrend="constant")
    m = (f >= band[0]) & (f <= band[1])
    return float(f[m][np.argmax(pxx[m])] * 60.0)
