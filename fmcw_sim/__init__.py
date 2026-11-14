"""X-band FMCW radar simulator: signal synthesis, range-Doppler processing, GUI."""

from .config import Impairments, ProcessingConfig, RadarConfig, Scenario, Target
from .presets import PRESETS
from .processing import expected_snr_db, range_doppler
from .signal import simulate_frame

__version__ = "0.1.0"
__all__ = [
    "Impairments", "ProcessingConfig", "RadarConfig", "Scenario", "Target",
    "PRESETS", "expected_snr_db", "range_doppler", "simulate_frame",
]
