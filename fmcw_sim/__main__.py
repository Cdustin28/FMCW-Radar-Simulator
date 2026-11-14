"""Entry point: ``python -m fmcw_sim`` opens the GUI; ``--headless`` runs one frame."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import io
from .presets import PRESETS
from .processing import expected_snr_db, range_doppler
from .signal import simulate_frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fmcw-sim", description="X-band FMCW radar simulator")
    ap.add_argument("--preset", choices=list(PRESETS), help="start from a built-in preset")
    ap.add_argument("--scenario", help="start from a saved scenario JSON file")
    ap.add_argument("--headless", action="store_true", help="simulate one frame and print detections")
    ap.add_argument("--export", metavar="FILE.npz", help="with --headless, save the I/Q frame")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --headless")
    args = ap.parse_args(argv)

    if args.scenario:
        scenario = io.load_scenario(args.scenario)
    elif args.preset:
        scenario = PRESETS[args.preset]()
    else:
        scenario = None

    if not args.headless:
        from .gui import run
        return run(scenario)

    scenario = scenario or next(iter(PRESETS.values()))()
    r = scenario.radar
    iq = simulate_frame(r, scenario.targets, scenario.impairments, np.random.default_rng(args.seed))
    rd = range_doppler(iq, r, scenario.processing)
    print(f"{scenario.name}: {iq.shape[0]} chirps x {iq.shape[1]} samples")
    print(f"range res {r.range_resolution_m:.2f} m, max range {r.max_range_m:.0f} m, "
          f"velocity res {r.velocity_resolution_mps:.3f} m/s, max velocity +/-{r.max_velocity_mps:.2f} m/s")
    for t in scenario.targets:
        print(f"target  {t.range_m:9.1f} m {t.velocity_mps:7.2f} m/s  expected SNR "
              f"{expected_snr_db(r, t, scenario.processing):5.1f} dB")
    for d in rd.detections:
        print(f"detect  {d.range_m:9.1f} m {d.velocity_mps:7.2f} m/s  {d.power_dbm:7.1f} dBm")
    if args.export:
        io.export_iq(args.export, iq, scenario)
        print(f"saved {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
