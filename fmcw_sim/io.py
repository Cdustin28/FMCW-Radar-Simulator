"""Scenario files (JSON) and raw I/Q export (NumPy .npz)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Scenario

FORMAT_VERSION = 1


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    data = {"format_version": FORMAT_VERSION, **scenario.to_dict()}
    Path(path).write_text(json.dumps(data, indent=2))


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.from_dict(json.loads(Path(path).read_text()))


def export_iq(path: str | Path, iq: np.ndarray, scenario: Scenario) -> None:
    """Save a frame as ``iq[n_chirps, n_samples]`` plus axes and the scenario.

    Load with ``np.load(path)``; ``scenario_json`` restores via
    ``Scenario.from_dict(json.loads(str(d["scenario_json"])))``.
    """
    r = scenario.radar
    np.savez_compressed(
        path,
        iq=iq.astype(np.complex64),
        fast_time_s=np.arange(iq.shape[1]) / r.fs_hz,
        slow_time_s=np.arange(iq.shape[0]) * r.prt_s,
        fs_hz=r.fs_hz,
        scenario_json=json.dumps(scenario.to_dict()),
    )


def load_iq(path: str | Path) -> tuple[np.ndarray, Scenario]:
    with np.load(path) as d:
        return d["iq"], Scenario.from_dict(json.loads(str(d["scenario_json"])))
