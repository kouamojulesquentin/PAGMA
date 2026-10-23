"""Stimulus generators (PWL): DC, PRBS, APRMLS (8 levels), STEP, SINUS, CHIRP, MULTISINE.

All stimuli in a circuit share the same observation window and the same number
of points, making the datasets comparable. The frequencies of the analog signals
are derived from the output sampling rate (no aliasing by design).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

STIMULI = ("DC", "PRBS", "APRMLS", "STEP", "SINUS", "CHIRP", "MULTISINE")
PRBS_PERIOD = 127 

def prbs7():

    state = [1] * 7
    seq = []
    for _ in range(PRBS_PERIOD):
        seq.append(state[-1])
        state = [state[6] ^ state[5]] + state[:-1]
    return seq


@dataclass
class StimConfig:
    v_low: float
    v_high: float
    bit_time: float            # PRBS/APRMLS symbol duration
    window: float              # duration of the observation window
    n_points: int              # number of points in the output vector
    t_sig_start: float = 0.0   # Start of valid signal
    t_obs_start: float = 0.0   # Start of the observation window
    v_init: Optional[float] = None      # pre-signal level (default: v_low)
    startup_from_zero: bool = False     # ramp from 0 to v_init in 1 µs (power supplies)
    rise: Optional[float] = None        # rise/fall time (default: bit_time/50)
    n_levels: int = 8                   # APRMLS Levels
    dc_window: Optional[float] = None   # If defined, the “DC” stimulus is allowed
    dc_points: Optional[int] = None
    sine_freq: Optional[float] = None
    chirp_band: Optional[tuple] = None
    multisine_freqs: Optional[tuple] = None

    @property
    def t_stop(self) -> float:
        return self.t_obs_start + self.window

    @property
    def tr(self) -> float:
        return self.rise if self.rise else self.bit_time / 50

    @property
    def v_mid(self) -> float:
        return 0.5 * (self.v_low + self.v_high)

    @property
    def amp(self) -> float:
        return 0.5 * (self.v_high - self.v_low)

    @property
    def fs_out(self) -> float:
        return (self.n_points - 1) / self.window


@dataclass
class Stimulus:
    name: str
    points: list = field(default_factory=list) 
    t_obs_start: float = 0.0
    t_stop: float = 0.0
    n_points: int = 0

    @property
    def pwl(self) -> Optional[str]:
        if not self.points:
            return None
        return " ".join(f"{t:.10g} {v:.6g}" for t, v in self.points)

    def grid(self) -> np.ndarray:
        return np.linspace(self.t_obs_start, self.t_stop, self.n_points)

    def sample_input(self) -> Optional[np.ndarray]:
        if not self.points:
            return None
        p = np.asarray(self.points)
        return np.interp(self.grid(), p[:, 0], p[:, 1])


def _prefix(cfg: StimConfig):
    v0 = cfg.v_init if cfg.v_init is not None else cfg.v_low
    pts = []
    if cfg.startup_from_zero:
        pts += [(0.0, 0.0), (1e-6, v0)]
    else:
        pts.append((0.0, v0))
    if cfg.t_sig_start > pts[-1][0]:
        pts.append((cfg.t_sig_start, v0))
    return pts, v0


def _symbols(cfg: StimConfig, level_fn):
    pts, cur = _prefix(cfg)
    k = 0
    while cfg.t_sig_start + k * cfg.bit_time < cfg.t_stop:
        lvl = level_fn(k)
        t = cfg.t_sig_start + k * cfg.bit_time
        if lvl != cur:
            pts.append((t + cfg.tr, lvl))
            cur = lvl
        pts.append((t + cfg.bit_time, cur))
        k += 1
    return pts


def _sampled(cfg: StimConfig, fn, f_max: float):
    dt = 1.0 / (32 * f_max)
    pts, _ = _prefix(cfg)
    t0 = cfg.t_sig_start + cfg.tr
    tau = np.arange(0.0, cfg.t_stop - t0 + dt, dt)
    vals = cfg.v_mid + cfg.amp * fn(tau)
    pts += [(t0 + a, float(v)) for a, v in zip(tau, vals)]
    return pts


def _check_nyquist(cfg: StimConfig, f_max: float, what: str):
    if f_max > 0.4 * cfg.fs_out:
        raise ValueError(
            f"{what}: f_max={f_max:.3g} Hz > 0.4*fs_out={0.4*cfg.fs_out:.3g} Hz "
            "(aliasing). Increase n_points or reduce the frequency."
        )


def make_stimulus(kind: str, cfg: StimConfig) -> Stimulus:
    kind = kind.upper()
    if kind not in STIMULI:
        raise ValueError(f"Unknown stimulus: {kind}. Choices: {STIMULI}")

    if kind == "DC":
        if not cfg.dc_window:
            raise ValueError("The DC stimulus is not defined for this circuit.")
        return Stimulus("DC", [], 0.0, cfg.dc_window, cfg.dc_points)

    if kind in ("PRBS", "APRMLS") and cfg.bit_time < 2 * cfg.window / (cfg.n_points - 1):
        raise ValueError("bit_time < 2 sampling periods: PRBS subsampling.")

    fs = cfg.fs_out
    bits = prbs7()
    if kind == "PRBS":
        pts = _symbols(cfg, lambda k: cfg.v_high if bits[k % PRBS_PERIOD] else cfg.v_low)

    elif kind == "APRMLS":

        m = int(round(np.log2(cfg.n_levels)))
        if 2 ** m != cfg.n_levels:
            raise ValueError("n_levels must be a power of 2 (default is 8).")
        shifts = [round(i * PRBS_PERIOD / m) for i in range(m)]
        levels = np.linspace(cfg.v_low, cfg.v_high, cfg.n_levels)

        def lvl(k):
            idx = sum(bits[(k + s) % PRBS_PERIOD] << i for i, s in enumerate(shifts))
            return float(levels[idx])

        pts = _symbols(cfg, lvl)

    elif kind == "STEP":
        t_step = cfg.t_sig_start + cfg.window / 3
        pts, _ = _prefix(cfg)
        pts += [(cfg.t_sig_start + cfg.tr, cfg.v_low), (t_step, cfg.v_low),
                (t_step + cfg.tr, cfg.v_high), (cfg.t_stop, cfg.v_high)]

    elif kind == "SINUS":
        f = cfg.sine_freq or 0.05 * fs
        _check_nyquist(cfg, f, "SINUS")
        pts = _sampled(cfg, lambda tau: np.sin(2 * np.pi * f * tau), f)

    elif kind == "CHIRP":
        f0, f1 = cfg.chirp_band or (4.0 / cfg.window, 0.2 * fs)
        _check_nyquist(cfg, f1, "CHIRP")
        T = cfg.t_stop - cfg.t_sig_start
        k = (f1 - f0) / T
        pts = _sampled(cfg, lambda tau: np.sin(2 * np.pi * (f0 * tau + 0.5 * k * tau ** 2)), f1)

    else:  
        freqs = cfg.multisine_freqs or tuple(fs * x for x in (0.01, 0.02, 0.05, 0.1, 0.2))
        _check_nyquist(cfg, max(freqs), "MULTISINE")
        K = len(freqs)
        phases = [-np.pi * i * (i - 1) / K for i in range(1, K + 1)]

        def fn(tau):
            s = sum(np.sin(2 * np.pi * f * tau + p) for f, p in zip(freqs, phases))
            return s / np.max(np.abs(s))

        pts = _sampled(cfg, fn, max(freqs))

    t = np.array([p[0] for p in pts])
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{kind}: PWL moments that are not strictly increasing.")
    return Stimulus(kind, pts, cfg.t_obs_start, cfg.t_stop, cfg.n_points)
