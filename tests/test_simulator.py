"""Checks the simulated range-Doppler output against closed-form predictions."""

import math

import numpy as np
import pytest

from fmcw_sim import (
    PRESETS, Impairments, ProcessingConfig, RadarConfig, Scenario, Target,
    expected_snr_db, range_doppler, simulate_frame,
)
from fmcw_sim import io
from fmcw_sim.config import C, K_B, T0, db_to_lin
from fmcw_sim.processing import cfar_threshold_factor, make_window
from fmcw_sim.signal import (
    nonlinearity_phase, phase_noise_difference, phase_noise_variance_rate,
    quantize, swerling_rcs,
)


@pytest.fixture
def radar():
    # Small frame so the suite runs quickly: 15 m resolution, 3.84 km, +/-17 m/s.
    return RadarConfig(fc_hz=9.6e9, bandwidth_hz=10e6, chirp_time_s=400e-6,
                       idle_time_s=50e-6, fs_hz=1.28e6, n_chirps=64)


def peak(rd):
    i_d, i_r = np.unravel_index(np.argmax(rd.power_w), rd.power_w.shape)
    return rd.range_m[i_r], rd.velocity_mps[i_d], rd.power_w[i_d, i_r]


NOISELESS = dict(thermal_noise=False)


# ------------------------------------------------------------ derived values
def test_derived_parameters(radar):
    lam = C / 9.6e9
    assert radar.n_samples == 512
    assert radar.range_resolution_m == pytest.approx(C / 2e7)
    assert radar.slope_hz_per_s == pytest.approx(2.5e10)
    assert radar.max_range_m == pytest.approx(C * 0.64e6 / (2 * 2.5e10))
    assert radar.max_velocity_mps == pytest.approx(lam / (4 * 450e-6))
    assert radar.velocity_resolution_mps == pytest.approx(lam / (2 * 64 * 450e-6))
    assert radar.noise_power_w == pytest.approx(K_B * T0 * db_to_lin(5.0) * 1.28e6)


def test_presets_are_valid():
    for make in PRESETS.values():
        sc = make()
        assert sc.radar.problems() == []
        for t in sc.targets:
            assert t.range_m < sc.radar.max_range_m
            assert abs(t.velocity_mps) < sc.radar.max_velocity_mps


# ----------------------------------------------------------- range / Doppler
@pytest.mark.parametrize("rng_m,vel", [(500.0, 0.0), (1500.0, 6.0), (3000.0, -11.0)])
def test_peak_location(radar, rng_m, vel):
    proc = ProcessingConfig(range_pad=4, doppler_pad=4, cfar=False)
    iq = simulate_frame(radar, [Target(rng_m, vel, 10.0)], Impairments(**NOISELESS))
    r, v, _ = peak(range_doppler(iq, radar, proc))
    # Doppler adds fd to the beat frequency (range-Doppler coupling), and the
    # target moves v*T_frame/2 by mid-frame.
    coupling = (2 * vel / radar.wavelength_m) * C / (2 * radar.slope_hz_per_s)
    expected_r = rng_m + coupling + vel * radar.frame_time_s / 2
    assert r == pytest.approx(expected_r, abs=radar.range_resolution_m / 2)
    assert v == pytest.approx(vel, abs=radar.velocity_resolution_mps / 2)


def test_peak_power_matches_radar_equation(radar):
    # Put the target on exact FFT bins so there is no straddle loss.
    bin_r, bin_d = 100, 5
    rng_m = bin_r * radar.fs_hz / radar.n_samples * C / (2 * radar.slope_hz_per_s)
    proc = ProcessingConfig(range_window="rect", doppler_window="rect",
                            range_pad=1, doppler_pad=1, cfar=False)
    iq = simulate_frame(radar, [Target(rng_m, 0.0, 3.0)], Impairments(**NOISELESS))
    _, _, p = peak(range_doppler(iq, radar, proc))
    assert 10 * math.log10(p / radar.received_power_w(rng_m, db_to_lin(3.0))) == pytest.approx(0.0, abs=0.05)


def test_two_targets_resolved(radar):
    dr = radar.range_resolution_m
    proc = ProcessingConfig(range_window="rect", cfar=False, range_pad=8, doppler_pad=1)
    iq = simulate_frame(radar, [Target(1000.0, 0, 10), Target(1000.0 + 3 * dr, 0, 10)],
                        Impairments(**NOISELESS))
    prof = range_doppler(iq, radar, proc).power_w.max(axis=0)
    rd_axis = range_doppler(iq, radar, proc).range_m
    seg = prof[(rd_axis > 1000.0) & (rd_axis < 1000.0 + 3 * dr)]
    assert seg.min() < 0.5 * prof.max()  # a clear dip between the two peaks


# ---------------------------------------------------------------- noise / SNR
@pytest.mark.parametrize("win", ["rect", "hann", "blackman"])
def test_noise_floor(radar, win):
    proc = ProcessingConfig(range_window=win, doppler_window=win, cfar=False)
    rd = range_doppler(simulate_frame(radar, [], Impairments(), np.random.default_rng(1)), radar, proc)
    assert 10 * math.log10(rd.power_w.mean() / rd.noise_floor_w) == pytest.approx(0.0, abs=0.2)


def test_snr_matches_prediction(radar):
    proc = ProcessingConfig(range_pad=4, doppler_pad=4, cfar=False)
    tgt = Target(2000.0, 4.0, 0.0)
    rng = np.random.default_rng(2)
    rd = range_doppler(simulate_frame(radar, [tgt], Impairments(), rng), radar, proc)
    noise = range_doppler(simulate_frame(radar, [], Impairments(), rng), radar, proc).power_w.mean()
    measured = 10 * math.log10(peak(rd)[2] / noise)
    assert measured == pytest.approx(expected_snr_db(radar, tgt, proc), abs=1.0)


def test_snr_follows_r4(radar):
    proc = ProcessingConfig()
    a = expected_snr_db(radar, Target(1000.0, 0, 0), proc)
    b = expected_snr_db(radar, Target(2000.0, 0, 0), proc)
    assert a - b == pytest.approx(40 * math.log10(2.0), abs=1e-9)


# ----------------------------------------------------------------------- CFAR
def test_cfar_finds_targets(radar):
    tgts = [Target(800.0, 5.0, 5.0), Target(2500.0, -9.0, 15.0)]
    rd = range_doppler(simulate_frame(radar, tgts, Impairments(), np.random.default_rng(3)),
                       radar, ProcessingConfig())
    for t in tgts:
        assert any(abs(d.range_m - t.range_m) < 2 * radar.range_resolution_m
                   and abs(d.velocity_mps - t.velocity_mps) < 2 * radar.velocity_resolution_mps
                   for d in rd.detections)
    assert len(rd.detections) <= len(tgts) + 2


def test_cfar_false_alarm_rate(radar):
    proc = ProcessingConfig(cfar_pfa=1e-3, doppler_pad=1, range_window="rect", doppler_window="rect")
    rng = np.random.default_rng(4)
    n = sum(len(range_doppler(simulate_frame(radar, [], Impairments(), rng), radar, proc).detections)
            for _ in range(10))
    cells = 10 * radar.n_chirps * radar.n_samples // 2
    # Local-maximum grouping can only lower the count below Pfa * cells.
    assert 0.2 * 1e-3 * cells < n < 1.5 * 1e-3 * cells


def test_cfar_threshold_factor_limit():
    # As the training count grows the factor tends to -ln(Pfa).
    assert cfar_threshold_factor(100000, 1e-6) == pytest.approx(-math.log(1e-6), rel=1e-3)


# ---------------------------------------------------------------- impairments
def test_phase_noise_difference_variance():
    imp = Impairments(phase_noise=True, phase_noise_dbc_hz=-70.0, phase_noise_offset_hz=100e3)
    rate = phase_noise_variance_rate(imp)
    assert rate == pytest.approx(4 * math.pi**2 * 1e-7 * 1e10)
    rng = np.random.default_rng(5)
    for tau in (0.3e-6, 2.7e-6):  # shorter and longer than one sample period
        d = phase_noise_difference(rng, rate, tau, 200, 400, 1e6)
        assert d.shape == (200, 400)
        assert d.var() == pytest.approx(rate * tau, rel=0.05)


def test_phase_noise_grows_with_range(radar):
    imp = Impairments(phase_noise=True, phase_noise_dbc_hz=-60.0, **NOISELESS)
    proc = ProcessingConfig(cfar=False)

    def skirt(rng_m):
        # Same echo power at both ranges, so only the delay differs.
        rcs = 40 * math.log10(rng_m / 300.0)
        rd = range_doppler(simulate_frame(radar, [Target(rng_m, 0, rcs)], imp,
                                          np.random.default_rng(6)), radar, proc)
        return np.median(rd.power_w) / rd.power_w.max()

    assert skirt(3000.0) > 5 * skirt(300.0)  # range-correlation effect


def test_nonlinearity_reduces_peak(radar):
    proc = ProcessingConfig(cfar=False)
    tgt = [Target(3000.0, 0, 10)]
    clean = peak(range_doppler(simulate_frame(radar, tgt, Impairments(**NOISELESS)), radar, proc))[2]
    imp = Impairments(nonlinearity=True, nonlin_bow_pct=2.0, **NOISELESS)
    bent = peak(range_doppler(simulate_frame(radar, tgt, imp), radar, proc))[2]
    assert bent < 0.7 * clean


def test_nonlinearity_phase_is_integral_of_frequency_error(radar):
    imp = Impairments(nonlin_bow_pct=1.0, nonlin_ripple_pct=0.3, nonlin_ripple_cycles=4)
    t = np.linspace(0, radar.chirp_time_s, 20001)
    f_err = np.gradient(nonlinearity_phase(t, radar, imp), t) / (2 * math.pi)
    x = t / radar.chirp_time_s
    expect = radar.bandwidth_hz * (0.01 * 4 * x * (1 - x) + 0.003 * np.sin(2 * math.pi * 4 * x))
    assert np.max(np.abs(f_err - expect)[5:-5]) < 1e-3 * radar.bandwidth_hz * 0.01


def test_quantizer_sqnr():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, 200000) + 1j * rng.uniform(-1, 1, 200000)
    for bits in (6, 10):
        err = quantize(x, bits, 0.0) - x
        sqnr = 10 * math.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(err) ** 2))
        assert sqnr == pytest.approx(6.02 * bits, abs=0.3)  # uniform full-scale input


def test_leakage_then_adc_and_filter(radar):
    tgt = [Target(2000.0, 3.0, 10.0)]
    proc = ProcessingConfig(cfar=False)
    base = dict(leakage=True, isolation_db=30.0, adc=True, adc_bits=6)
    rng = np.random.default_rng(8)
    raw = range_doppler(simulate_frame(radar, tgt, Impairments(**base), rng), radar, proc)
    filt = range_doppler(simulate_frame(radar, tgt, Impairments(**base, if_filter=True, hpf_cutoff_hz=20e3),
                                        rng), radar, proc)
    # Leakage sets the ADC full scale; removing it before the ADC restores the floor.
    assert np.median(raw.power_w) > 10 * np.median(filt.power_w)


def test_interference_raises_floor(radar):
    proc = ProcessingConfig(cfar=False)
    rng = np.random.default_rng(9)
    clean = range_doppler(simulate_frame(radar, [], Impairments(), rng), radar, proc)
    imp = Impairments(interference=True, interf_range_m=1000.0, interf_bandwidth_hz=15e6)
    dirty = range_doppler(simulate_frame(radar, [], imp, rng), radar, proc)
    assert np.median(dirty.power_w) > 4 * np.median(clean.power_w)


def test_clutter_is_zero_doppler_and_mti_removes_it(radar):
    imp = Impairments(clutter=True, clutter_count=50, clutter_max_range_m=3000.0, **NOISELESS)
    iq = simulate_frame(radar, [], imp)
    rd = range_doppler(iq, radar, ProcessingConfig(cfar=False, doppler_pad=1))
    assert peak(rd)[1] == pytest.approx(0.0, abs=1e-9)
    mti = range_doppler(iq, radar, ProcessingConfig(cfar=False, doppler_pad=1, mti=True))
    assert mti.power_w.max() < 1e-12 * rd.power_w.max()


def test_multipath_lobing(radar):
    # With rho = -1 the two-way factor is |1 - exp(-j*2*pi*dR/lambda)|^2.
    lam = radar.wavelength_m
    h_r, rng_m = 10.0, 1000.0
    proc = ProcessingConfig(cfar=False, range_window="rect", doppler_window="rect")
    free = peak(range_doppler(simulate_frame(radar, [Target(rng_m, 0, 10)], Impairments(**NOISELESS)),
                              radar, proc))[2]
    for frac, gain in ((0.5, 16.0), (1.0, 0.0)):  # path difference of lambda/2 and lambda
        h_t = ((rng_m + frac * lam) ** 2 - rng_m**2) / (4 * h_r)
        imp = Impairments(multipath=True, radar_height_m=h_r, target_height_m=h_t,
                          reflection_coeff=-1.0, **NOISELESS)
        p = peak(range_doppler(simulate_frame(radar, [Target(rng_m, 0, 10)], imp), radar, proc))[2]
        assert p / free == pytest.approx(gain, abs=0.05)


@pytest.mark.parametrize("model,var_ratio", [(1, 1.0), (2, 1.0), (3, 0.5), (4, 0.5)])
def test_swerling_statistics(model, var_ratio):
    rng = np.random.default_rng(10)
    draws = np.concatenate([swerling_rcs(rng, 2.0, model, 64)[:, 0] for _ in range(4000)])
    assert draws.mean() == pytest.approx(2.0, rel=0.05)
    assert draws.var() / draws.mean() ** 2 == pytest.approx(var_ratio, rel=0.1)
    per_frame = swerling_rcs(rng, 2.0, model, 64)[:, 0]
    assert (np.ptp(per_frame) == 0) == (model in (1, 3))  # scan-to-scan vs pulse-to-pulse


# ------------------------------------------------------------------------ I/O
def test_scenario_round_trip(tmp_path):
    sc = PRESETS["Marine navigation"]()
    sc.impairments.phase_noise = True
    path = tmp_path / "s.json"
    io.save_scenario(sc, path)
    assert io.load_scenario(path) == sc


def test_iq_export_round_trip(tmp_path, radar):
    sc = Scenario(radar=radar, targets=[Target(900.0, 2.0, 5.0)])
    iq = simulate_frame(radar, sc.targets, sc.impairments, np.random.default_rng(11))
    path = tmp_path / "f.npz"
    io.export_iq(path, iq, sc)
    iq2, sc2 = io.load_iq(path)
    assert sc2 == sc
    assert np.allclose(iq2, iq.astype(np.complex64))


def test_window_names():
    from fmcw_sim.config import WINDOWS
    for name in WINDOWS:
        assert make_window(name, 64).shape == (64,)
