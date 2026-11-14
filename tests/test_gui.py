"""Smoke test for the GUI using Qt's offscreen platform."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from fmcw_sim.gui import MainWindow  # noqa: E402
from fmcw_sim.presets import PRESETS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    pg.setConfigOptions(background="w", foreground="k", imageAxisOrder="row-major")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_window_runs_all_presets(app):
    win = MainWindow()
    for make in PRESETS.values():
        win.set_scenario(make())
        assert win.last_iq.shape == (win.scenario.radar.n_chirps, win.scenario.radar.n_samples)
        assert win.target_table.rowCount() == len(win.scenario.targets)
    win.close()


def test_edit_animate_reset(app):
    win = MainWindow(PRESETS["Medium surveillance"]())
    win.target_table.item(0, 0).setText("2000")
    assert win.scenario.targets[0].range_m == 2000.0
    v = win.scenario.targets[0].velocity_mps
    win._advance_once()
    assert win.scenario.targets[0].range_m == pytest.approx(2000.0 + v * win.step_box.value())
    win.reset_scene()
    assert win.scenario.targets[0].range_m == 2000.0
    n = len(win.scenario.targets)
    win.add_target()
    assert win.target_table.rowCount() == n + 1
    win.close()


def test_forms_write_through(app):
    win = MainWindow()
    form = win.imp_form
    for attr, box in form._boxes:
        box.setChecked(True)
        assert getattr(win.scenario.impairments, attr) is True
    win.run_frame()  # every impairment enabled at once
    assert win.last_iq is not None
    win.close()
