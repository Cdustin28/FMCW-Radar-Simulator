"""X-band presets. Representative parameter sets, not models of specific products."""

from __future__ import annotations

from .config import Impairments, ProcessingConfig, RadarConfig, Scenario, Target


def short_range() -> Scenario:
    """0-600 m sensor: 0.6 m resolution, +/-24 m/s."""
    return Scenario(
        name="Short-range sensor (10.5 GHz)",
        radar=RadarConfig(
            fc_hz=10.5e9, bandwidth_hz=250e6, chirp_time_s=250e-6, idle_time_s=50e-6,
            fs_hz=8e6, n_chirps=128, tx_power_w=0.1, antenna_gain_dbi=15.0,
            noise_figure_db=8.0, system_loss_db=3.0,
        ),
        targets=[
            Target(60.0, -4.0, 0.0, 0),
            Target(150.0, 12.0, 10.0, 1),
            Target(320.0, -18.0, 10.0, 0),
        ],
        impairments=Impairments(
            hpf_cutoff_hz=20e3, radar_height_m=1.5, target_height_m=1.0,
            clutter_mean_rcs_dbsm=-5.0, clutter_min_range_m=5.0,
            interf_range_m=500.0, interf_power_w=0.1,
            interf_bandwidth_hz=200e6, interf_fc_offset_hz=20e6,
            interf_chirp_time_s=180e-6,
        ),
    )


def surveillance() -> Scenario:
    """0-12 km surveillance: 7.5 m resolution, +/-17 m/s."""
    return Scenario(
        name="Medium surveillance (9.6 GHz)",
        radar=RadarConfig(),
        targets=[
            Target(1500.0, 9.0, -15.0, 1),   # small drone
            Target(4200.0, -12.0, 10.0, 1),  # vehicle
            Target(8000.0, 5.0, 15.0, 3),    # small boat
        ],
        impairments=Impairments(),
    )


def marine() -> Scenario:
    """0-30 km marine navigation: 15 m resolution, sea clutter on by default."""
    return Scenario(
        name="Marine navigation (9.4 GHz)",
        radar=RadarConfig(
            fc_hz=9.4e9, bandwidth_hz=10e6, chirp_time_s=1e-3, idle_time_s=100e-6,
            fs_hz=4e6, n_chirps=64, tx_power_w=2.0, antenna_gain_dbi=28.0,
            noise_figure_db=5.0, system_loss_db=4.0,
        ),
        targets=[
            Target(2500.0, 3.0, 20.0, 1),    # small craft
            Target(9000.0, -5.0, 35.0, 1),   # coaster
            Target(18000.0, 2.0, 45.0, 3),   # large vessel
        ],
        impairments=Impairments(
            clutter=True, clutter_count=200, clutter_mean_rcs_dbsm=5.0,
            clutter_min_range_m=100.0, clutter_max_range_m=6000.0,
            clutter_velocity_spread_mps=0.4, hpf_cutoff_hz=2e3,
            radar_height_m=15.0, target_height_m=5.0, reflection_coeff=-0.6,
            interf_range_m=5000.0, interf_power_w=2.0,
            interf_bandwidth_hz=15e6, interf_fc_offset_hz=1e6,
            interf_chirp_time_s=700e-6,
        ),
        processing=ProcessingConfig(cfar_train_range=10),
    )


PRESETS = {
    "Short-range sensor": short_range,
    "Medium surveillance": surveillance,
    "Marine navigation": marine,
}
