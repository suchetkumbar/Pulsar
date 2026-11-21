"""Temporal aliasing of light-source modulation into the camera's sampled signal.

A periodic illumination component at f Hz sampled at fs frames/s appears at
    f_alias = | f - fs * round(f / fs) |   (in [0, fs/2]).
Used to explain (docs 04 §4.2) why mains flicker and ceiling-fan harmonics can land inside the
heart-rate band when phones drop frame rate in low light.
"""

from __future__ import annotations

import numpy as np

from pulsar.signal.filters import HR_BAND_HZ


def alias_frequency(f_hz: float | np.ndarray, fs: float) -> np.ndarray | float:
    f = np.asarray(f_hz, dtype=np.float64)
    out = np.abs(f - fs * np.round(f / fs))
    return float(out) if out.ndim == 0 else out


def fan_blade_pass_hz(rpm: float, blades: int) -> float:
    return blades * rpm / 60.0


def in_hr_band(f_hz: float, band: tuple[float, float] = HR_BAND_HZ) -> bool:
    return band[0] <= f_hz <= band[1]


DEFAULT_FPS = (30.0, 29.97, 25.0, 24.5, 24.0, 20.0, 15.0)


def default_sources() -> list[tuple[str, float]]:
    src = [("Mains flicker 100 Hz", 100.0), ("Flicker at grid 49.9 Hz", 99.8)]
    for blades, rpm in [(3, 350), (3, 250), (4, 300)]:
        fb = fan_blade_pass_hz(rpm, blades)
        src.append((f"Fan {blades}-blade {rpm} rpm", fb))
        for n in (2, 3):
            src.append((f"  harmonic {n}", n * fb))
    return src


def alias_table_markdown(
    sources: list[tuple[str, float]] | None = None, fps: tuple[float, ...] = DEFAULT_FPS
) -> str:
    sources = default_sources() if sources is None else sources
    head = "| Source | f (Hz) | " + " | ".join(f"{f:g} fps" for f in fps) + " |"
    sep = "|" + "---|" * (2 + len(fps))
    rows = [head, sep]
    for name, f in sources:
        cells = []
        for fs in fps:
            a = alias_frequency(f, fs)
            s = f"{a:.2f}"
            cells.append(f"**{s}**" if in_hr_band(a) else s)
        rows.append(f"| {name} | {f:g} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
