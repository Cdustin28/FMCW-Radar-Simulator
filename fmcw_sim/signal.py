"""Dechirped (beat) signal synthesis for a sawtooth FMCW frame.

Model
-----
Transmit phase within a chirp (fast time t):

    phi_tx(t) = 2*pi*(fc*t + S*t^2/2) + eps(t) + pn(t)

where ``eps`` is the deterministic sweep-nonlinearity phase and ``pn`` the
oscillator phase noise. An echo with round-trip delay tau mixes with the
transmit signal (tx * conj(rx)) to give

    phi_b(t) = 2*pi*(fc*tau + S*tau*t - S*tau^2/2)
               + [eps(t) - eps(t - tau)] + [pn(t) - pn(t - tau)]

tau varies with slow and fast time as the target moves, which produces the
Doppler term. Positive velocity means increasing range, giving a positive
beat-frequency shift under this mixing convention.

Signals are scaled so that |x|^2 is power in watts at the receiver input.
"""

from __future__ import annotations

import math

import numpy as np

from .config import C, Impairments, RadarConfig, Target, db_to_lin

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- impairments
def nonlinearity_phase(t: np.ndarray, radar: RadarConfig, imp: Impairments) -> np.ndarray:
    """Transmit phase error eps(t) in radians from sweep nonlinearity."""
    b, tc = radar.bandwidth_hz, radar.chirp_time_s
    bow = imp.nonlin_bow_pct / 100.0 * b
    # frequency error 4*bow*(t/Tc)*(1 - t/Tc), integrated
    eps = TWO_PI * 4.0 * bow * (t**2 / (2.0 * tc) - t**3 / (3.0 * tc**2))
    k = imp.nonlin_ripple_cycles
    if imp.nonlin_ripple_pct != 0.0 and k > 0:
        rip = imp.nonlin_ripple_pct / 100.0 * b
        # frequency error rip*sin(2*pi*k*t/Tc), integrated
        eps = eps - rip * tc / k * np.cos(TWO_PI * k * t / tc)
    return eps


def phase_noise_variance_rate(imp: Impairments) -> float:
    """Wiener phase-noise diffusion in rad^2/s.

    For L(f) = L0*(f0/f)^2 the phase is a random walk with
    Var[phi(t+tau) - phi(t)] = 4*pi^2*L0*f0^2*tau.
    """
    return 4.0 * math.pi**2 * db_to_lin(imp.phase_noise_dbc_hz) * imp.phase_noise_offset_hz**2


def phase_noise_difference(
    rng: np.random.Generator, var_rate: float, tau: float, n_chirps: int,
    n_samples: int, fs: float,
) -> np.ndarray:
    """Exact samples of pn(t) - pn(t - tau) for a Wiener phase process.

    The walk is generated on the union of the sample grid and the grid delayed
    by tau, so the range-correlation effect (short delays cancel close-in phase
    noise) is reproduced without oversampling.
    """
    ts = 1.0 / fs
    k = int(math.floor(tau / ts))
    r = tau - k * ts
    n = n_samples + k
    inc_a = rng.normal(0.0, math.sqrt(var_rate * max(ts - r, 0.0)), (n_chirps, n))
    inc_b = rng.normal(0.0, math.sqrt(var_rate * r), (n_chirps, n))
    w_b = np.cumsum(inc_a + inc_b, axis=1)  # walk on the sample grid
    w_a = w_b - inc_b  # walk on the delayed grid
    return w_b[:, k:] - w_a[:, :n_samples]


def if_gain(f_hz, radar: RadarConfig, imp: Impairments):
    """Butterworth band-pass magnitude at beat frequency ``f_hz``."""
    f = np.abs(np.asarray(f_hz, dtype=float))
    g = np.ones_like(f)
    if not imp.if_filter:
        return g
    f_lp = imp.lpf_cutoff_frac * radar.fs_hz / 2.0
    if f_lp > 0:
        g = g / np.sqrt(1.0 + (f / f_lp) ** (2 * imp.lpf_order))
    if imp.hpf_cutoff_hz > 0:
        with np.errstate(divide="ignore"):
            ratio = np.where(f > 0, imp.hpf_cutoff_hz / np.maximum(f, 1e-30), np.inf)
        g = g / np.sqrt(1.0 + ratio ** (2 * imp.hpf_order))
    return g


def swerling_rcs(rng: np.random.Generator, mean_rcs: float, model: int, n_chirps: int) -> np.ndarray:
    """RCS per chirp, shape (n_chirps, 1), for Swerling models 0-4."""
    if model in (1, 2):
        size = 1 if model == 1 else n_chirps
        rcs = rng.exponential(mean_rcs, size)
    elif model in (3, 4):
        size = 1 if model == 3 else n_chirps
        rcs = rng.gamma(2.0, mean_rcs / 2.0, size)
    else:
        rcs = np.array([mean_rcs])
    return np.broadcast_to(rcs, (n_chirps,)).reshape(n_chirps, 1)


def quantize(x: np.ndarray, bits: int, headroom_db: float) -> np.ndarray:
    """Mid-rise quantiser on I and Q with full scale set from the frame peak."""
    peak = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))
    if peak <= 0:
        return x
    full_scale = peak * 10.0 ** (headroom_db / 20.0)
    step = 2.0 * full_scale / (2**bits)
    lim = full_scale - step / 2.0

    def q(v):
        return np.clip((np.floor(v / step) + 0.5) * step, -lim, lim)

    return q(x.real) + 1j * q(x.imag)


# ------------------------------------------------------------------ synthesis
class _Frame:
    def __init__(self, radar: RadarConfig, imp: Impairments, rng: np.random.Generator):
        self.radar, self.imp, self.rng = radar, imp, rng
        self.m, self.n = radar.n_chirps, radar.n_samples
        self.t_fast = np.arange(self.n) / radar.fs_hz  # (N,)
        self.t_slow = (np.arange(self.m) * radar.prt_s).reshape(-1, 1)  # (M,1)
        self.t_abs = self.t_slow + self.t_fast  # (M,N)
        self.iq = np.zeros((self.m, self.n), dtype=np.complex128)
        self.eps = nonlinearity_phase(self.t_fast, radar, imp) if imp.nonlinearity else None
        self.pn_rate = phase_noise_variance_rate(imp) if imp.phase_noise else 0.0

    def add_return(self, path_m: float, path_rate_mps: float, power_w, sign: float = 1.0) -> None:
        """Add one echo. ``path_m`` is the total (out + back) path at frame start."""
        radar, imp = self.radar, self.imp
        s = radar.slope_hz_per_s
        tau = (path_m + path_rate_mps * self.t_abs) / C
        phase = TWO_PI * (radar.fc_hz * tau + s * tau * self.t_fast - 0.5 * s * tau**2)
        tau0 = path_m / C
        if self.eps is not None:
            phase += self.eps - nonlinearity_phase(self.t_fast - tau0, radar, imp)
        if self.pn_rate > 0.0 and tau0 > 0.0:
            phase += phase_noise_difference(self.rng, self.pn_rate, tau0, self.m, self.n, radar.fs_hz)
        f_beat = s * tau0 + radar.fc_hz * path_rate_mps / C
        amp = sign * np.sqrt(power_w) * if_gain(f_beat, radar, imp)
        self.iq += amp * np.exp(1j * phase)


def _path_power(radar: RadarConfig, r_out: float, r_back: float, rcs) -> np.ndarray:
    g = db_to_lin(radar.antenna_gain_dbi)
    return (
        radar.tx_power_w * g * g * radar.wavelength_m**2 * rcs
        / ((4.0 * math.pi) ** 3 * r_out**2 * r_back**2 * db_to_lin(radar.system_loss_db))
    )


def _add_targets(fr: _Frame, targets: list[Target]) -> None:
    radar, imp = fr.radar, fr.imp
    for tgt in targets:
        if tgt.range_m <= 0:
            continue
        rcs = swerling_rcs(fr.rng, db_to_lin(tgt.rcs_dbsm), tgt.swerling, fr.m)
        r_d = tgt.range_m
        rate = 2.0 * tgt.velocity_mps
        fr.add_return(2.0 * r_d, rate, _path_power(radar, r_d, r_d, rcs))
        if imp.multipath:
            # Two-ray model: image path is longer by ~2*h_r*h_t/R.
            r_i = math.sqrt(r_d**2 + 4.0 * imp.radar_height_m * imp.target_height_m)
            rho = imp.reflection_coeff
            # direct-out/bounce-back and bounce-out/direct-back are coherent and equal
            p_mixed = _path_power(radar, r_d, r_i, rcs) * (2.0 * rho) ** 2
            p_double = _path_power(radar, r_i, r_i, rcs) * rho**4
            fr.add_return(r_d + r_i, rate, p_mixed, sign=1.0 if rho >= 0 else -1.0)
            fr.add_return(2.0 * r_i, rate, p_double)


def _add_clutter(fr: _Frame, t0: float) -> None:
    """Clutter scatterers, using a separable fast-time x slow-time model."""
    radar, imp = fr.radar, fr.imp
    k = int(imp.clutter_count)
    if k <= 0:
        return
    crng = np.random.default_rng(imp.clutter_seed)  # fixed scene across frames
    r_max = imp.clutter_max_range_m if imp.clutter_max_range_m > 0 else radar.max_range_m
    r_min = min(max(imp.clutter_min_range_m, 1.0), r_max)
    rng_m = crng.uniform(r_min, r_max, k)
    rcs = crng.exponential(db_to_lin(imp.clutter_mean_rcs_dbsm), k)
    vel = crng.normal(0.0, 1.0, k) * imp.clutter_velocity_spread_mps
    phi0 = crng.uniform(0.0, TWO_PI, k)

    s = radar.slope_hz_per_s
    tau = (2.0 * rng_m / C).reshape(-1, 1)  # (K,1)
    phase = TWO_PI * (s * tau * fr.t_fast - 0.5 * s * tau**2) + phi0.reshape(-1, 1)
    if fr.eps is not None:
        phase += fr.eps - nonlinearity_phase(fr.t_fast - tau, radar, imp)
    f_beat = s * tau[:, 0] + 2.0 * vel / radar.wavelength_m
    amp = np.sqrt(_path_power(radar, rng_m, rng_m, rcs)) * if_gain(f_beat, radar, imp)
    fast = amp.reshape(-1, 1) * np.exp(1j * phase)  # (K,N)
    slow_t = t0 + fr.t_slow  # (M,1)
    slow = np.exp(1j * TWO_PI * 2.0 * vel.reshape(1, -1) / radar.wavelength_m * slow_t)  # (M,K)
    fr.iq += slow @ fast


def _add_interference(fr: _Frame, t0: float) -> None:
    radar, imp = fr.radar, fr.imp
    t_i = imp.interf_chirp_time_s + imp.interf_idle_time_s
    if t_i <= 0 or imp.interf_chirp_time_s <= 0 or imp.interf_range_m <= 0:
        return
    s_i = imp.interf_bandwidth_hz / imp.interf_chirp_time_s
    t = fr.t_abs + t0
    idx = np.floor(t / t_i).astype(np.int64)
    u = t - idx * t_i
    active = u < imp.interf_chirp_time_s
    # random start phase per interferer chirp (incoherent with the victim)
    idx0 = idx.min()
    theta = fr.rng.uniform(0.0, TWO_PI, int(idx.max() - idx0) + 1)[idx - idx0]
    s = radar.slope_hz_per_s
    f_inst = s * fr.t_fast - imp.interf_fc_offset_hz - s_i * u
    phase = TWO_PI * (0.5 * s * fr.t_fast**2 - imp.interf_fc_offset_hz * u - 0.5 * s_i * u**2) + theta
    power = (
        imp.interf_power_w * db_to_lin(imp.interf_gain_dbi) * db_to_lin(imp.interf_rx_gain_dbi)
        * radar.wavelength_m**2 / ((4.0 * math.pi) ** 2 * imp.interf_range_m**2)
    )
    fr.iq += active * math.sqrt(power) * if_gain(f_inst, radar, imp) * np.exp(1j * phase)


def simulate_frame(
    radar: RadarConfig,
    targets: list[Target],
    imp: Impairments | None = None,
    rng: np.random.Generator | None = None,
    t0: float = 0.0,
) -> np.ndarray:
    """Simulate one frame of dechirped I/Q data.

    Returns a complex array of shape ``(n_chirps, n_samples)`` in sqrt(W).
    ``t0`` is the scene time at the start of the frame; it only affects the
    clutter Doppler phase and the interferer timing.
    """
    imp = imp or Impairments()
    rng = rng or np.random.default_rng()
    fr = _Frame(radar, imp, rng)

    _add_targets(fr, targets)
    if imp.leakage:
        fr.add_return(imp.leakage_path_m, 0.0, radar.tx_power_w / db_to_lin(imp.isolation_db))
    if imp.clutter:
        _add_clutter(fr, t0)
    if imp.interference:
        _add_interference(fr, t0)
    if imp.thermal_noise:
        sigma = math.sqrt(radar.noise_power_w / 2.0)
        fr.iq += sigma * (rng.standard_normal(fr.iq.shape) + 1j * rng.standard_normal(fr.iq.shape))
    if imp.adc:
        fr.iq = quantize(fr.iq, int(imp.adc_bits), imp.adc_headroom_db)
    return fr.iq
