"""Configuration dataclasses for the FMCW radar simulator.

All quantities are SI unless the field name says otherwise (``_db``, ``_dbi``,
``_dbsm``, ``_pct``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, asdict

C = 299_792_458.0  # speed of light, m/s
K_B = 1.380649e-23  # Boltzmann constant, J/K
T0 = 290.0  # reference noise temperature, K

WINDOWS = ("rect", "hann", "hamming", "blackman", "chebyshev 80 dB")


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def lin_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-300))


def _from_dict(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys (forward compatible)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class RadarConfig:
    """Sawtooth FMCW radar with complex (I/Q) dechirp sampling."""

    fc_hz: float = 9.6e9  # chirp start frequency
    bandwidth_hz: float = 20e6  # swept bandwidth B
    chirp_time_s: float = 400e-6  # sweep duration Tc
    idle_time_s: float = 50e-6  # dead time between chirps
    fs_hz: float = 8e6  # complex IF sample rate
    n_chirps: int = 128  # chirps per frame (slow-time samples)
    tx_power_w: float = 10.0
    antenna_gain_dbi: float = 30.0  # same antenna gain on transmit and receive
    noise_figure_db: float = 5.0
    system_loss_db: float = 3.0

    # ---- derived quantities -------------------------------------------------
    @property
    def wavelength_m(self) -> float:
        return C / self.fc_hz

    @property
    def slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_time_s

    @property
    def prt_s(self) -> float:
        """Chirp repetition interval."""
        return self.chirp_time_s + self.idle_time_s

    @property
    def n_samples(self) -> int:
        """Fast-time samples per chirp."""
        return max(8, int(math.floor(self.chirp_time_s * self.fs_hz + 1e-9)))

    @property
    def frame_time_s(self) -> float:
        return self.n_chirps * self.prt_s

    @property
    def range_resolution_m(self) -> float:
        return C / (2.0 * self.bandwidth_hz)

    @property
    def max_beat_hz(self) -> float:
        """Highest unaliased beat frequency (positive half of the complex band)."""
        return self.fs_hz / 2.0

    @property
    def max_range_m(self) -> float:
        """Range whose beat frequency equals fs/2."""
        return C * self.max_beat_hz / (2.0 * self.slope_hz_per_s)

    @property
    def velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.frame_time_s)

    @property
    def max_velocity_mps(self) -> float:
        """Unambiguous radial velocity is +/- this value."""
        return self.wavelength_m / (4.0 * self.prt_s)

    @property
    def noise_power_w(self) -> float:
        """Thermal noise power in the sampled bandwidth, k*T0*F*fs."""
        return K_B * T0 * db_to_lin(self.noise_figure_db) * self.fs_hz

    def beat_frequency_hz(self, range_m: float, velocity_mps: float = 0.0) -> float:
        """Beat frequency for a target (positive velocity = receding)."""
        return (
            self.slope_hz_per_s * 2.0 * range_m / C
            + 2.0 * velocity_mps / self.wavelength_m
        )

    def received_power_w(self, range_m: float, rcs_m2: float) -> float:
        """Monostatic radar range equation (mean power for mean RCS)."""
        g = db_to_lin(self.antenna_gain_dbi)
        return (
            self.tx_power_w * g * g * self.wavelength_m**2 * rcs_m2
            / ((4.0 * math.pi) ** 3 * range_m**4 * db_to_lin(self.system_loss_db))
        )

    def problems(self) -> list[str]:
        """Human-readable list of configuration problems (empty when valid)."""
        out = []
        if self.bandwidth_hz <= 0 or self.chirp_time_s <= 0 or self.fs_hz <= 0:
            out.append("Bandwidth, chirp time and sample rate must be positive.")
        if self.n_chirps < 2:
            out.append("At least 2 chirps per frame are required.")
        if self.chirp_time_s * self.fs_hz * self.n_chirps > 4e6:
            out.append("Frame exceeds 4 M samples; reduce fs, chirp time or chirps.")
        if self.bandwidth_hz > 0.2 * self.fc_hz:
            out.append("Bandwidth exceeds 20 % of the carrier.")
        return out


@dataclass
class Target:
    range_m: float = 1000.0
    velocity_mps: float = 0.0  # range rate; positive = receding
    rcs_dbsm: float = 0.0
    swerling: int = 0  # 0 (non-fluctuating) to 4


@dataclass
class Impairments:
    thermal_noise: bool = True

    # Oscillator phase noise: 1/f^2 (Wiener) profile through the given point.
    phase_noise: bool = False
    phase_noise_dbc_hz: float = -85.0
    phase_noise_offset_hz: float = 100e3

    # Sweep nonlinearity, as peak frequency error in percent of B.
    nonlinearity: bool = False
    nonlin_bow_pct: float = 0.5  # quadratic bow, zero at sweep ends
    nonlin_ripple_pct: float = 0.0  # sinusoidal ripple
    nonlin_ripple_cycles: float = 3.0  # ripple periods across the sweep

    # Static / slow clutter as random point scatterers.
    clutter: bool = False
    clutter_count: int = 150
    clutter_mean_rcs_dbsm: float = 0.0
    clutter_min_range_m: float = 50.0
    clutter_max_range_m: float = 0.0  # 0 = radar maximum range
    clutter_velocity_spread_mps: float = 0.0
    clutter_seed: int = 1

    # Two-ray ground / sea bounce.
    multipath: bool = False
    radar_height_m: float = 10.0
    target_height_m: float = 5.0
    reflection_coeff: float = -0.7

    # Transmit-to-receive leakage.
    leakage: bool = False
    isolation_db: float = 60.0
    leakage_path_m: float = 0.3

    # IF filter (analytic Butterworth magnitude response).
    if_filter: bool = False
    hpf_cutoff_hz: float = 10e3
    hpf_order: int = 2
    lpf_cutoff_frac: float = 0.9  # fraction of fs/2
    lpf_order: int = 6

    # ADC quantisation with automatic full-scale.
    adc: bool = False
    adc_bits: int = 12
    adc_headroom_db: float = 3.0  # negative values clip

    # Interference from another FMCW radar (one-way path).
    interference: bool = False
    interf_power_w: float = 1.0
    interf_gain_dbi: float = 0.0  # interferer antenna gain toward this radar
    interf_rx_gain_dbi: float = 0.0  # this radar's gain toward the interferer (sidelobe)
    interf_range_m: float = 2000.0
    interf_fc_offset_hz: float = 2e6
    interf_bandwidth_hz: float = 30e6
    interf_chirp_time_s: float = 250e-6
    interf_idle_time_s: float = 30e-6


@dataclass
class ProcessingConfig:
    range_window: str = "hann"
    doppler_window: str = "hann"
    range_pad: int = 1  # zero-padding factor
    doppler_pad: int = 2
    mti: bool = False  # subtract slow-time mean (removes zero-Doppler)
    cfar: bool = True
    cfar_pfa: float = 1e-6
    cfar_guard_range: int = 2
    cfar_guard_doppler: int = 2
    cfar_train_range: int = 8
    cfar_train_doppler: int = 4


@dataclass
class Scenario:
    name: str = "Custom"
    radar: RadarConfig = field(default_factory=RadarConfig)
    targets: list[Target] = field(default_factory=list)
    impairments: Impairments = field(default_factory=Impairments)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        return cls(
            name=data.get("name", "Custom"),
            radar=_from_dict(RadarConfig, data.get("radar", {})),
            targets=[_from_dict(Target, t) for t in data.get("targets", [])],
            impairments=_from_dict(Impairments, data.get("impairments", {})),
            processing=_from_dict(ProcessingConfig, data.get("processing", {})),
        )
