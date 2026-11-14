"""Range-Doppler processing, CFAR detection and analytic predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal as sps

from .config import C, ProcessingConfig, RadarConfig, Target, db_to_lin, lin_to_db


def make_window(name: str, n: int) -> np.ndarray:
    if name == "rect" or n < 3:
        return np.ones(n)
    if name == "hann":
        return sps.windows.hann(n, sym=False)
    if name == "hamming":
        return sps.windows.hamming(n, sym=False)
    if name == "blackman":
        return sps.windows.blackman(n, sym=False)
    if name.startswith("cheb"):
        return sps.windows.chebwin(n, 80.0, sym=False)
    raise ValueError(f"unknown window '{name}'")


def window_noise_factor(w: np.ndarray) -> float:
    """sum(w^2)/sum(w)^2: noise power per bin after a coherent-gain-normalised FFT."""
    return float(np.sum(w**2) / np.sum(w) ** 2)


@dataclass
class Detection:
    range_m: float
    velocity_mps: float
    power_dbm: float
    snr_db: float


@dataclass
class RangeDopplerMap:
    power_w: np.ndarray  # (n_doppler, n_range); a tone of power P peaks at P
    range_m: np.ndarray
    velocity_mps: np.ndarray
    noise_floor_w: float  # expected thermal noise per cell
    detections: list[Detection] = field(default_factory=list)

    @property
    def power_dbm(self) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.power_w, 1e-300)) + 30.0


def expected_noise_floor_w(radar: RadarConfig, proc: ProcessingConfig) -> float:
    wr = make_window(proc.range_window, radar.n_samples)
    wd = make_window(proc.doppler_window, radar.n_chirps)
    return radar.noise_power_w * window_noise_factor(wr) * window_noise_factor(wd)


def expected_snr_db(radar: RadarConfig, target: Target, proc: ProcessingConfig) -> float:
    """Predicted range-Doppler cell SNR for the mean RCS (no straddle loss)."""
    if target.range_m <= 0:
        return float("nan")
    pr = radar.received_power_w(target.range_m, db_to_lin(target.rcs_dbsm))
    return lin_to_db(pr / expected_noise_floor_w(radar, proc))


def range_doppler(iq: np.ndarray, radar: RadarConfig, proc: ProcessingConfig) -> RangeDopplerMap:
    """Windowed 2-D FFT of a frame, scaled so a tone of power P reads P watts."""
    m, n = iq.shape
    x = iq - iq.mean(axis=0, keepdims=True) if proc.mti else iq
    wr = make_window(proc.range_window, n)
    wd = make_window(proc.doppler_window, m)
    n_fft = int(n * max(1, proc.range_pad))
    m_fft = int(m * max(1, proc.doppler_pad))

    rng_spec = np.fft.fft(x * wr, n=n_fft, axis=1)[:, : n_fft // 2] / wr.sum()
    rd = np.fft.fftshift(np.fft.fft(rng_spec * wd[:, None], n=m_fft, axis=0), axes=0) / wd.sum()

    beat = np.arange(n_fft // 2) * radar.fs_hz / n_fft
    range_m = C * beat / (2.0 * radar.slope_hz_per_s)
    doppler = np.fft.fftshift(np.fft.fftfreq(m_fft, radar.prt_s))
    velocity = doppler * radar.wavelength_m / 2.0

    out = RangeDopplerMap(np.abs(rd) ** 2, range_m, velocity, expected_noise_floor_w(radar, proc))
    if proc.cfar:
        out.detections = ca_cfar(out, proc)
    return out


def cfar_threshold_factor(n_train: int, pfa: float) -> float:
    """Cell-averaging CFAR multiplier for square-law data."""
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def ca_cfar(rd: RangeDopplerMap, proc: ProcessingConfig, max_detections: int = 64) -> list[Detection]:
    """2-D cell-averaging CFAR followed by local-maximum grouping.

    Guard and training sizes are specified in resolution cells and scaled by
    the zero-padding factors. The Doppler axis wraps; range edges reflect.
    """
    p = rd.power_w
    gr = proc.cfar_guard_range * max(1, proc.range_pad)
    gd = proc.cfar_guard_doppler * max(1, proc.doppler_pad)
    tr = max(1, proc.cfar_train_range) * max(1, proc.range_pad)
    td = max(1, proc.cfar_train_doppler) * max(1, proc.doppler_pad)
    outer = (2 * (gd + td) + 1, 2 * (gr + tr) + 1)
    inner = (2 * gd + 1, 2 * gr + 1)
    if outer[0] > p.shape[0]:
        outer = (p.shape[0] - (1 - p.shape[0] % 2), outer[1])
        inner = (min(inner[0], max(1, outer[0] - 2)), inner[1])
    mode = ("wrap", "reflect")
    n_outer, n_inner = outer[0] * outer[1], inner[0] * inner[1]
    n_train = n_outer - n_inner
    s_outer = ndimage.uniform_filter(p, outer, mode=mode) * n_outer
    s_inner = ndimage.uniform_filter(p, inner, mode=mode) * n_inner
    noise = np.maximum(s_outer - s_inner, 0.0) / n_train

    # Zero padding correlates neighbouring cells, so use the independent count.
    n_indep = max(4, n_train // (max(1, proc.range_pad) * max(1, proc.doppler_pad)))
    thresh = cfar_threshold_factor(n_indep, proc.cfar_pfa) * noise
    peak = p == ndimage.maximum_filter(
        p, size=(2 * max(1, proc.doppler_pad) + 1, 2 * max(1, proc.range_pad) + 1), mode=mode
    )
    hits = np.argwhere((p > thresh) & peak)
    if hits.size == 0:
        return []
    order = np.argsort(p[hits[:, 0], hits[:, 1]])[::-1][:max_detections]
    out = []
    for i_d, i_r in hits[order]:
        out.append(Detection(
            float(rd.range_m[i_r]), float(rd.velocity_mps[i_d]),
            lin_to_db(p[i_d, i_r]) + 30.0,
            lin_to_db(p[i_d, i_r] / max(noise[i_d, i_r], 1e-300)),
        ))
    return out


def range_profile_dbm(rd: RangeDopplerMap) -> np.ndarray:
    """Peak over Doppler for each range bin, in dBm."""
    return 10.0 * np.log10(np.maximum(rd.power_w.max(axis=0), 1e-300)) + 30.0


def beat_spectrogram(chirp: np.ndarray, fs: float):
    """Spectrogram of one dechirped chirp. Returns (freq_hz, time_s, power_db)."""
    n = len(chirp)
    nper = int(min(256, max(16, 2 ** int(math.log2(max(n // 8, 16))))))
    f, t, sxx = sps.spectrogram(
        chirp, fs=fs, window="hann", nperseg=nper, noverlap=nper * 3 // 4,
        return_onesided=False, scaling="spectrum", mode="psd",
    )
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)
    return f, t, 10.0 * np.log10(np.maximum(sxx, 1e-300)) + 30.0
