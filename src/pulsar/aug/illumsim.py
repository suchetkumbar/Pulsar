"""IllumSim-IN: simulator of Indian indoor illumination artifacts at trace level (docs 04 §4).

Operates on a window of per-patch linear-RGB traces x [T, P, 3] on a uniform fs grid, plus each
patch's normalised image centre (x, y) and pixel count. Pipeline:

    1. draw a capture frame rate (incl. low-light drops), jitter and frame drops
    2. interpolate the traces to those capture instants
    3. multiply by the illumination field G(t_n, x_k, y_k) evaluated *at the capture instants*
       (mains flicker w/ rolling shutter + exposure sinc, ceiling-fan shadow harmonics, window
       light drift/clouds/gradient, colour-temperature mix)
    4. camera auto-exposure loop (global gain with lag, dead-band, quantised steps)
    5. sensor-noise top-up for the new intensity
    6. resample back to the fs grid (exactly what the deployed pipeline does)

Aliasing therefore emerges physically; `pulsar.signal.alias` predicts where it lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from pulsar.aug.sensor_noise import DEFAULT_A, DEFAULT_B, topup_noise

LUMA = np.array([0.2126, 0.7152, 0.0722])

# illuminant RGB gains relative to daylight (before partial AWB correction); approximate
ILLUMINANTS = {
    "daylight": np.array([1.00, 1.00, 1.00]),
    "cool_tube": np.array([0.92, 1.05, 0.98]),  # fluorescent tube light ~6500 K, green spike
    "warm_led": np.array([1.25, 1.00, 0.70]),  # ~3000 K
    "incandescent": np.array([1.35, 1.00, 0.60]),  # ~2700 K
}


@dataclass(frozen=True)
class IllumSimConfig:
    # component probabilities (training); evaluation presets set them to 0/1
    p_flicker: float = 0.5
    p_fan: float = 0.5
    p_window: float = 0.5
    p_cct: float = 0.5
    p_ae: float = 0.5
    p_fps: float = 0.5
    # flicker
    flicker_depth: tuple[float, float] = (0.0, 0.4)
    mains_hz: float = 50.0
    grid_drift: float = 0.003
    exposures_s: tuple[float, ...] = (1 / 30, 1 / 50, 1 / 60, 1 / 100, 1 / 120, 1 / 200)
    readout_s: tuple[float, float] = (0.010, 0.030)
    # fan
    fan_depth: tuple[float, float] = (0.0, 0.10)
    fan_rpm: tuple[float, float] = (120.0, 400.0)
    fan_blades: tuple[int, ...] = (3, 4)
    fan_chirp_prob: float = 0.3
    # window light
    window_sigma: float = 0.10
    window_tau_s: tuple[float, float] = (5.0, 30.0)
    cloud_rate_hz: float = 0.05
    cloud_amp: tuple[float, float] = (-0.5, 0.3)
    window_gradient: float = 0.3
    # colour temperature
    awb_strength: tuple[float, float] = (0.6, 0.95)
    # auto exposure
    ae_rate: tuple[float, float] = (0.05, 0.3)
    ae_deadband: float = 0.03
    ae_step: float = 2 ** (1 / 6)
    # frame timing
    capture_fps: tuple[float, ...] = (30.0, 29.97, 25.0, 24.5, 24.0, 20.0)
    jitter_ms: tuple[float, float] = (0.5, 3.0)
    drop_frac: tuple[float, float] = (0.0, 0.02)
    # sensor noise
    noise_a: float = DEFAULT_A
    noise_b: float = DEFAULT_B
    extra: dict = field(default_factory=dict)


def severity_preset(level: int) -> IllumSimConfig:
    """Evaluation presets S0-S4 (docs 04 §4.4). S4 is never used for training."""
    base = IllumSimConfig(p_flicker=1, p_fan=1, p_window=1, p_cct=1, p_ae=1, p_fps=1)
    presets = {
        0: replace(base, p_flicker=0, p_fan=0, p_window=0, p_cct=0, p_ae=0, p_fps=0),
        1: replace(
            base,
            flicker_depth=(0.1, 0.1),
            fan_depth=(0.02, 0.02),
            window_sigma=0.03,
            cloud_rate_hz=0.0,
            p_cct=0,
            ae_rate=(0.05, 0.05),
            capture_fps=(30.0, 29.97),
            drop_frac=(0.0, 0.0),
        ),
        2: replace(
            base,
            flicker_depth=(0.25, 0.25),
            fan_depth=(0.05, 0.05),
            window_sigma=0.06,
            cloud_rate_hz=0.1,
            ae_rate=(0.15, 0.15),
            capture_fps=(30.0, 29.97, 25.0),
            drop_frac=(0.0, 0.01),
        ),
        3: replace(
            base,
            flicker_depth=(0.4, 0.4),
            fan_depth=(0.10, 0.10),
            window_sigma=0.10,
            cloud_rate_hz=0.15,
            ae_rate=(0.2, 0.2),
            drop_frac=(0.0, 0.02),
            capture_fps=(25.0, 24.5, 24.0, 20.0),
        ),
        4: replace(
            base,
            flicker_depth=(0.6, 0.6),
            fan_depth=(0.15, 0.15),
            window_sigma=0.15,
            cloud_rate_hz=0.2,
            ae_rate=(0.3, 0.3),
            capture_fps=(24.5, 20.0, 15.0),
            drop_frac=(0.03, 0.03),
        ),
    }
    return presets[level]


@dataclass
class SimInfo:
    capture_fps: float
    components: list[str]
    params: dict


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(lo if lo == hi else rng.uniform(lo, hi))


def capture_times(t_grid: np.ndarray, fps: float, jitter_s: float, drop: float, rng) -> np.ndarray:
    t0, t1 = t_grid[0], t_grid[-1]
    n = int(np.floor((t1 - t0) * fps)) + 1
    t = t0 + np.arange(n) / fps + rng.uniform(0, 1 / fps)
    t = t + rng.normal(0, jitter_s, n)
    if drop > 0:
        keep = rng.random(n) > drop
        t = t[keep]
    t = np.sort(t[(t >= t0) & (t <= t1)])
    # guarantee coverage of the window ends for interpolation back to the grid
    return np.unique(np.concatenate([[t0], t, [t1]]))


def _interp_time(t_new: np.ndarray, t_old: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0 for x [T, ...]."""
    idx = np.clip(np.searchsorted(t_old, t_new) - 1, 0, len(t_old) - 2)
    w = (t_new - t_old[idx]) / (t_old[idx + 1] - t_old[idx])
    w = w.reshape((-1,) + (1,) * (x.ndim - 1))
    return x[idx] * (1 - w) + x[idx + 1] * w


def flicker_field(t, y, rng, cfg: IllumSimConfig, params: dict) -> np.ndarray:
    """1 + m |sinc(f T_exp)| cos(2 pi f (t + y tau) + phi_c), shape [T, P, 3]."""
    f = 2 * cfg.mains_hz * (1 + rng.uniform(-cfg.grid_drift, cfg.grid_drift))
    m = _u(rng, cfg.flicker_depth)
    t_exp = float(rng.choice(cfg.exposures_s))
    tau = _u(rng, cfg.readout_s)
    att = abs(np.sinc(f * t_exp))  # numpy sinc is sin(pi x)/(pi x)
    phi = rng.uniform(0, 2 * np.pi) + rng.normal(0, 0.15, 3)  # phosphor per-channel spread
    amp = m * att * rng.uniform(0.85, 1.15, 3)
    params.update(flicker_hz=f, flicker_depth=m, exposure_s=t_exp, readout_s=tau, flicker_att=att)
    arg = 2 * np.pi * f * (t[:, None] + y[None, :] * tau)  # [T, P]
    return 1 + amp[None, None, :] * np.cos(arg[..., None] + phi[None, None, :])


def fan_field(t, y, rng, cfg: IllumSimConfig, params: dict) -> np.ndarray:
    blades = int(rng.choice(cfg.fan_blades))
    rpm0 = _u(rng, cfg.fan_rpm)
    d = _u(rng, cfg.fan_depth)
    if rng.random() < cfg.fan_chirp_prob:  # spin-up / spin-down sweep
        rpm1 = _u(rng, cfg.fan_rpm)
        rpm = np.linspace(rpm0, rpm1, t.size)
    else:
        rpm = np.full(t.size, rpm0)
    fb = blades * rpm / 60.0
    phase = 2 * np.pi * np.concatenate([[0.0], np.cumsum(0.5 * (fb[1:] + fb[:-1]) * np.diff(t))])
    kappa = rng.uniform(0, 2 * np.pi)  # shadow sweeps across rows
    g = np.zeros((t.size, y.size))
    norm = 0.0
    for n in range(1, 5):
        a_n = 1.0 / n
        norm += a_n
        g += a_n * np.cos(n * phase[:, None] + rng.uniform(0, 2 * np.pi) + kappa * y[None, :])
    params.update(
        fan_blades=blades, fan_rpm=(float(rpm[0]), float(rpm[-1])), fan_depth=d, fan_bpf_hz=float(fb[0])
    )
    return (1 - d * g / norm)[..., None]


def window_field(t, xpos, rng, cfg: IllumSimConfig, params: dict) -> np.ndarray:
    tau = _u(rng, cfg.window_tau_s)
    dt = np.diff(t, prepend=t[0])
    ou = np.zeros(t.size)
    for i in range(1, t.size):
        ou[i] = (
            ou[i - 1]
            - ou[i - 1] * dt[i] / tau
            + cfg.window_sigma * np.sqrt(2 * dt[i] / tau) * rng.standard_normal()
        )
    # cloud events: smooth steps
    dur = t[-1] - t[0]
    n_cloud = rng.poisson(cfg.cloud_rate_hz * dur)
    level = np.zeros(t.size)
    for _ in range(n_cloud):
        tc = rng.uniform(t[0], t[-1])
        w = rng.uniform(0.5, 1.5)  # ~1-3 s transitions
        level += _u(rng, cfg.cloud_amp) / (1 + np.exp(-(t - tc) / w))
    intensity = np.exp(ou) * np.clip(1 + level, 0.2, None)
    gamma = cfg.window_gradient * np.sin(2 * np.pi * t / rng.uniform(20, 60) + rng.uniform(0, 6.3))
    params.update(window_tau_s=tau, n_clouds=int(n_cloud))
    g = intensity[:, None] * (1 + gamma[:, None] * (xpos[None, :] - 0.5))
    return g[..., None]


def cct_field(t, rng, cfg: IllumSimConfig, params: dict) -> np.ndarray:
    names = list(ILLUMINANTS)
    a, b = rng.choice(names, size=2, replace=True)
    beta = _u(rng, cfg.awb_strength)
    g1 = ILLUMINANTS[a] ** (1 - beta)
    g2 = ILLUMINANTS[b] ** (1 - beta)
    w = 0.5 + 0.5 * np.sin(2 * np.pi * t / rng.uniform(15, 60) + rng.uniform(0, 6.3))
    w = w * rng.uniform(0, 1)  # mix strength
    g = (1 - w)[:, None] * g1[None] + w[:, None] * g2[None]
    g = g / (g @ LUMA)[:, None]  # keep luminance, change chromaticity only
    params.update(illuminants=(str(a), str(b)), awb_strength=beta)
    return g[:, None, :]


def auto_exposure(x: np.ndarray, rng, cfg: IllumSimConfig, params: dict) -> np.ndarray:
    """Global AE loop: gain chases the initial luminance with lag, dead-band and ISO steps."""
    k = _u(rng, cfg.ae_rate)
    luma = (x @ LUMA).mean(axis=1)  # [T]
    target = luma[: max(3, len(luma) // 20)].mean()
    gains = np.ones(len(luma))
    g = 1.0
    for i in range(len(luma)):
        gains[i] = g
        err = target / max(g * luma[i], 1e-9) - 1
        if abs(err) > cfg.ae_deadband:
            g_new = g * (1 + k * err)
            steps = np.round(np.log(g_new / g) / np.log(cfg.ae_step))
            g = g * cfg.ae_step**steps if steps != 0 else g
    params.update(ae_rate=k)
    return gains


def simulate(
    x: np.ndarray,
    center: np.ndarray,
    n_pix: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    cfg: IllumSimConfig | None = None,
) -> tuple[np.ndarray, SimInfo]:
    """Apply IllumSim-IN to one window.

    Args:
        x: [T, P, 3] linear-RGB patch traces on a uniform grid at `fs`.
        center: [P, 2] normalised (x, y) patch centres (y = rolling-shutter row).
        n_pix: [P] pixels per patch (noise model).
    Returns:
        (x_aug [T, P, 3] on the same grid, SimInfo)
    """
    cfg = cfg or IllumSimConfig()
    t_len, p_n, _ = x.shape
    t_grid = np.arange(t_len) / fs
    params: dict = {}
    comps: list[str] = []

    fps = float(rng.choice(cfg.capture_fps)) if rng.random() < cfg.p_fps else fs
    jit = _u(rng, cfg.jitter_ms) / 1000.0 if fps != fs else 0.0
    drop = _u(rng, cfg.drop_frac) if fps != fs else 0.0
    if fps != fs:
        comps.append("fps")
        tc = capture_times(t_grid, fps, jit, drop, rng)
    else:
        tc = t_grid
    xc = _interp_time(tc, t_grid, x) if fps != fs else x.copy()
    xpos = np.nan_to_num(center[:, 0], nan=0.5)
    ypos = np.nan_to_num(center[:, 1], nan=0.5)

    g = np.ones_like(xc)
    if rng.random() < cfg.p_flicker:
        g = g * flicker_field(tc, ypos, rng, cfg, params)
        comps.append("flicker")
    if rng.random() < cfg.p_fan:
        g = g * fan_field(tc, ypos, rng, cfg, params)
        comps.append("fan")
    if rng.random() < cfg.p_window:
        g = g * window_field(tc, xpos, rng, cfg, params)
        comps.append("window")
    if rng.random() < cfg.p_cct:
        g = g * cct_field(tc, rng, cfg, params)
        comps.append("cct")
    y = xc * g
    if rng.random() < cfg.p_ae:
        ae = auto_exposure(y, rng, cfg, params)
        y = y * ae[:, None, None]
        g = g * ae[:, None, None]
        comps.append("ae")
    if comps:
        y = topup_noise(y, xc, g, n_pix, rng, cfg.noise_a, cfg.noise_b)
    out = _interp_time(t_grid, tc, y) if fps != fs else y
    return out.astype(np.float32), SimInfo(fps, comps, params)
