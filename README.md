# FMCW Radar Simulator

An X-band sawtooth FMCW radar simulator with a desktop GUI. It synthesises the
dechirped I/Q signal for a frame of chirps, forms a range-Doppler map, runs
CA-CFAR detection, and lets you switch individual hardware and environment
impairments on and off to see what each one does.

Python (NumPy/SciPy) with a PySide6 + pyqtgraph front end.

## Features

- **Presets:** short-range sensor (10.5 GHz, 0.6 m resolution), medium
  surveillance (9.6 GHz, 12 km), and marine navigation (9.4 GHz, 30 km). Every
  parameter stays editable. The presets are representative parameter sets, not
  models of specific products.
- **Processing:** windowed 2-D FFT, optional zero-Doppler removal, 2-D
  cell-averaging CFAR with local-maximum grouping.
- **Impairments, each toggleable:** thermal noise, oscillator phase noise,
  chirp nonlinearity, clutter, two-ray multipath, Tx-Rx leakage, IF filtering,
  ADC quantisation, and interference from a second FMCW radar.
- **Scene:** editable target table (range, radial velocity, RCS, Swerling 0-4),
  animated frames, scenario save/load (JSON), raw I/Q export (`.npz`).
- **Readouts:** range/velocity resolution, unambiguous range/velocity, noise
  power, processing gain, and predicted SNR per target, for checking the map
  against hand calculations.

## Install and run

Arch Linux, system packages:

```sh
sudo pacman -S python-numpy python-scipy pyside6 python-pyqtgraph python-pytest
python -m fmcw_sim
```

Or in a virtual environment on any platform:

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
fmcw-sim
```

Options: `--preset "Marine navigation"`, `--scenario file.json`, and
`--headless [--export frame.npz] [--seed N]` to simulate one frame without the
GUI and print the detections.

Run the tests with `pytest`. The GUI tests use Qt's offscreen platform and are
skipped when PySide6 is not installed.

## Using it as a library

```python
import numpy as np
from fmcw_sim import PRESETS, Target, simulate_frame, range_doppler

sc = PRESETS["Medium surveillance"]()
sc.targets = [Target(range_m=3000, velocity_mps=-8, rcs_dbsm=5)]
sc.impairments.phase_noise = True
iq = simulate_frame(sc.radar, sc.targets, sc.impairments, np.random.default_rng(0))
rd = range_doppler(iq, sc.radar, sc.processing)   # rd.power_w[doppler, range]
print(rd.detections)
```

## Signal model

With chirp slope `S = B/Tc` and round-trip delay `tau = 2R/c`, mixing the
transmit signal with the echo gives the beat phase

    phi_b(t) = 2*pi*(fc*tau + S*tau*t - S*tau^2/2)
               + [eps(t) - eps(t - tau)] + [pn(t) - pn(t - tau)]

where `eps` is the sweep-nonlinearity phase and `pn` the oscillator phase
noise. `tau` is evaluated at every fast- and slow-time sample, so Doppler,
range-Doppler coupling and range walk come out of the model rather than being
added separately. Positive velocity means increasing range.

Amplitudes follow the radar range equation, with `|x|^2` in watts at the
receiver input, and thermal noise is `k*T0*F*fs`. The range-Doppler map is
normalised by the window coherent gain, so a target reads its received power in
dBm directly, and

    SNR = Pr / (k*T0*F*fs * sum(w_r^2)/sum(w_r)^2 * sum(w_d^2)/sum(w_d)^2)

Key relations: range resolution `c/2B`; maximum range `c*fs/(4S)` (beat
frequency of `fs/2`); velocity resolution `lambda/(2*M*PRT)`; maximum velocity
`+/-lambda/(4*PRT)`.

### Impairment models and their limits

| Effect | Model | Simplification |
| --- | --- | --- |
| Phase noise | 1/f² (Wiener) phase walk through a specified dBc/Hz point; the difference `pn(t) - pn(t - tau)` is generated exactly, so close-in noise cancels at short range | Independent realisation per echo; not applied to clutter; no flat far-out floor |
| Chirp nonlinearity | Quadratic bow and sinusoidal ripple in the frequency sweep | Deterministic and identical every chirp |
| Clutter | Random point scatterers with exponential RCS and Gaussian velocity spread, fixed by a seed | Fast-time and slow-time treated as separable (no range walk) |
| Multipath | Two-ray flat-surface bounce: direct, mixed and double-bounce paths | One global target height; constant reflection coefficient |
| Leakage | An echo at a short fixed path with power set by the isolation | Single path |
| IF filter | Butterworth high-pass and low-pass magnitude applied to each echo at its beat frequency, and to the interferer at its instantaneous frequency | No phase response or transients; thermal noise is not shaped |
| ADC | Mid-rise quantiser on I and Q, full scale set from the frame peak plus headroom | No AGC dynamics or nonlinearity |
| Interference | A second sawtooth radar on a one-way path, random start phase per chirp | Single interferer, no multipath |

Swerling 1 and 3 draw one RCS per frame; 2 and 4 draw one per chirp.

## Layout

    fmcw_sim/config.py      dataclasses and derived radar parameters
    fmcw_sim/presets.py     the three X-band presets
    fmcw_sim/signal.py      I/Q synthesis and impairments
    fmcw_sim/processing.py  range-Doppler FFT, CFAR, SNR prediction
    fmcw_sim/io.py          scenario JSON and I/Q export
    fmcw_sim/gui.py         PySide6 + pyqtgraph application
    tests/                  checks against closed-form results

## License

MIT
