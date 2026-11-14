"""PySide6 + pyqtgraph front end for the FMCW radar simulator."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import numpy as np

# pyqtgraph binds to the first Qt wrapper it finds (PyQt6 before PySide6), and
# mixing two wrappers fails. Import PySide6 first and pin pyqtgraph to it.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import pyqtgraph as pg  # noqa: E402

from . import io
from .config import WINDOWS, Scenario, Target
from .presets import PRESETS
from .processing import (
    beat_spectrogram, expected_snr_db, range_doppler, range_profile_dbm,
)
from .signal import simulate_frame

CARDINAL = "#C8102E"
GOLD = "#F1BE48"
DARK_GOLD = "#9A7B1C"


# ------------------------------------------------------------ parameter forms
@dataclass
class P:
    """One editable field: attribute name, label, and display options."""

    attr: str
    label: str
    kind: str = "float"  # float | int | bool | choice
    lo: float = 0.0
    hi: float = 1e12
    scale: float = 1.0  # stored value = displayed value * scale
    unit: str = ""
    decimals: int = 3
    step: float = 1.0
    choices: tuple = ()


@dataclass
class G:
    """A group of fields, optionally switched by a boolean attribute."""

    title: str
    fields: list
    enable: str | None = None


class ParamForm(QtWidgets.QWidget):
    """Builds spin boxes / combos for a dataclass instance from a spec list."""

    changed = QtCore.Signal()

    def __init__(self, obj, groups: list[G]):
        super().__init__()
        self.obj = obj
        self._loading = False
        self._widgets: list[tuple[P, QtWidgets.QWidget]] = []
        self._boxes: list[tuple[str, QtWidgets.QGroupBox]] = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        for g in groups:
            box = QtWidgets.QGroupBox(g.title)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignRight)
            if g.enable:
                box.setCheckable(True)
                box.toggled.connect(lambda on, a=g.enable: self._set(a, bool(on)))
                self._boxes.append((g.enable, box))
            for p in g.fields:
                w = self._make(p)
                form.addRow(p.label, w)
                self._widgets.append((p, w))
            outer.addWidget(box)
        outer.addStretch(1)
        self.set_object(obj)

    def _make(self, p: P) -> QtWidgets.QWidget:
        if p.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.toggled.connect(lambda on, a=p.attr: self._set(a, bool(on)))
        elif p.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.currentIndexChanged.connect(lambda i, p=p: self._set(p.attr, p.choices[i]))
        elif p.kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(p.lo), int(p.hi))
            w.setSingleStep(max(1, int(p.step)))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, a=p.attr: self._set(a, int(v)))
        else:
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(p.decimals)
            w.setRange(p.lo, p.hi)
            w.setSingleStep(p.step)
            w.setKeyboardTracking(False)
            w.valueChanged.connect(lambda v, p=p: self._set(p.attr, float(v) * p.scale))
        if p.unit and hasattr(w, "setSuffix"):
            w.setSuffix(" " + p.unit)
        return w

    def _set(self, attr: str, value) -> None:
        if self._loading:
            return
        setattr(self.obj, attr, value)
        self.changed.emit()

    def set_object(self, obj) -> None:
        """Point the form at a new object and refresh every widget silently."""
        self.obj = obj
        self._loading = True
        try:
            for attr, box in self._boxes:
                box.setChecked(bool(getattr(obj, attr)))
            for p, w in self._widgets:
                v = getattr(obj, p.attr)
                if p.kind == "bool":
                    w.setChecked(bool(v))
                elif p.kind == "choice":
                    w.setCurrentIndex(p.choices.index(v) if v in p.choices else 0)
                elif p.kind == "int":
                    w.setValue(int(v))
                else:
                    w.setValue(float(v) / p.scale)
        finally:
            self._loading = False


RADAR_SPEC = [
    G("Waveform", [
        P("fc_hz", "Start frequency", lo=8.0, hi=12.5, scale=1e9, unit="GHz", decimals=4, step=0.05),
        P("bandwidth_hz", "Sweep bandwidth", lo=0.1, hi=2000.0, scale=1e6, unit="MHz", step=5.0),
        P("chirp_time_s", "Chirp duration", lo=1.0, hi=20000.0, scale=1e-6, unit="µs", decimals=1, step=50.0),
        P("idle_time_s", "Idle time", lo=0.0, hi=20000.0, scale=1e-6, unit="µs", decimals=1, step=10.0),
        P("n_chirps", "Chirps per frame", "int", lo=2, hi=1024, step=16),
    ]),
    G("Transmitter / receiver", [
        P("fs_hz", "IF sample rate (complex)", lo=0.01, hi=100.0, scale=1e6, unit="MHz", step=0.5),
        P("tx_power_w", "Transmit power", lo=0.0001, hi=10000.0, unit="W", decimals=4, step=0.5),
        P("antenna_gain_dbi", "Antenna gain", lo=0.0, hi=50.0, unit="dBi", decimals=1),
        P("noise_figure_db", "Noise figure", lo=0.0, hi=30.0, unit="dB", decimals=1, step=0.5),
        P("system_loss_db", "System losses", lo=0.0, hi=30.0, unit="dB", decimals=1, step=0.5),
    ]),
]

IMPAIRMENT_SPEC = [
    G("Thermal noise", [], enable="thermal_noise"),
    G("Oscillator phase noise (1/f² profile)", [
        P("phase_noise_dbc_hz", "Level", lo=-160.0, hi=-30.0, unit="dBc/Hz", decimals=1),
        P("phase_noise_offset_hz", "At offset", lo=0.1, hi=10000.0, scale=1e3, unit="kHz", decimals=1, step=10.0),
    ], enable="phase_noise"),
    G("Chirp nonlinearity", [
        P("nonlin_bow_pct", "Quadratic bow (peak)", lo=-20.0, hi=20.0, unit="% of B", step=0.1),
        P("nonlin_ripple_pct", "Sinusoidal ripple (peak)", lo=0.0, hi=20.0, unit="% of B", step=0.05),
        P("nonlin_ripple_cycles", "Ripple cycles per sweep", lo=0.1, hi=200.0, decimals=1),
    ], enable="nonlinearity"),
    G("Clutter", [
        P("clutter_count", "Scatterers", "int", lo=1, hi=2000, step=25),
        P("clutter_mean_rcs_dbsm", "Mean RCS", lo=-60.0, hi=60.0, unit="dBsm", decimals=1),
        P("clutter_min_range_m", "Min range", lo=1.0, hi=1e5, unit="m", decimals=0, step=50.0),
        P("clutter_max_range_m", "Max range (0 = radar max)", lo=0.0, hi=1e5, unit="m", decimals=0, step=250.0),
        P("clutter_velocity_spread_mps", "Velocity spread (1σ)", lo=0.0, hi=20.0, unit="m/s", decimals=2, step=0.1),
        P("clutter_seed", "Scene seed", "int", lo=0, hi=99999),
    ], enable="clutter"),
    G("Multipath (two-ray surface bounce)", [
        P("radar_height_m", "Radar height", lo=0.1, hi=1000.0, unit="m", decimals=1),
        P("target_height_m", "Target height", lo=0.1, hi=1000.0, unit="m", decimals=1),
        P("reflection_coeff", "Reflection coefficient", lo=-1.0, hi=1.0, decimals=2, step=0.05),
    ], enable="multipath"),
    G("Tx-Rx leakage", [
        P("isolation_db", "Isolation", lo=0.0, hi=150.0, unit="dB", decimals=1, step=5.0),
        P("leakage_path_m", "Leakage path length", lo=0.0, hi=100.0, unit="m", decimals=2, step=0.1),
    ], enable="leakage"),
    G("IF filter (Butterworth)", [
        P("hpf_cutoff_hz", "High-pass cutoff (0 = off)", lo=0.0, hi=10000.0, scale=1e3, unit="kHz", decimals=2),
        P("hpf_order", "High-pass order", "int", lo=1, hi=8),
        P("lpf_cutoff_frac", "Low-pass cutoff", lo=0.05, hi=2.0, unit="× fs/2", decimals=2, step=0.05),
        P("lpf_order", "Low-pass order", "int", lo=1, hi=12),
    ], enable="if_filter"),
    G("ADC quantisation (auto full-scale)", [
        P("adc_bits", "Resolution", "int", lo=1, hi=24, unit="bits"),
        P("adc_headroom_db", "Headroom (negative clips)", lo=-40.0, hi=40.0, unit="dB", decimals=1),
    ], enable="adc"),
    G("Interfering FMCW radar", [
        P("interf_power_w", "Transmit power", lo=0.0001, hi=10000.0, unit="W", decimals=4, step=0.5),
        P("interf_gain_dbi", "Its gain toward us", lo=-30.0, hi=50.0, unit="dBi", decimals=1),
        P("interf_rx_gain_dbi", "Our gain toward it", lo=-30.0, hi=50.0, unit="dBi", decimals=1),
        P("interf_range_m", "Distance", lo=1.0, hi=1e6, unit="m", decimals=0, step=100.0),
        P("interf_fc_offset_hz", "Start-frequency offset", lo=-2000.0, hi=2000.0, scale=1e6, unit="MHz", step=1.0),
        P("interf_bandwidth_hz", "Sweep bandwidth", lo=0.0, hi=2000.0, scale=1e6, unit="MHz", step=5.0),
        P("interf_chirp_time_s", "Chirp duration", lo=1.0, hi=20000.0, scale=1e-6, unit="µs", decimals=1, step=10.0),
        P("interf_idle_time_s", "Idle time", lo=0.0, hi=20000.0, scale=1e-6, unit="µs", decimals=1, step=10.0),
    ], enable="interference"),
]

PROCESSING_SPEC = [
    G("Range-Doppler FFT", [
        P("range_window", "Range window", "choice", choices=WINDOWS),
        P("doppler_window", "Doppler window", "choice", choices=WINDOWS),
        P("range_pad", "Range zero-pad factor", "int", lo=1, hi=8),
        P("doppler_pad", "Doppler zero-pad factor", "int", lo=1, hi=8),
        P("mti", "Remove zero-Doppler (MTI)", "bool"),
    ]),
    G("CA-CFAR detector", [
        P("cfar_pfa", "log10(Pfa)", lo=-12.0, hi=-1.0, decimals=1, step=0.5),
        P("cfar_guard_range", "Guard cells, range", "int", lo=0, hi=32),
        P("cfar_guard_doppler", "Guard cells, Doppler", "int", lo=0, hi=32),
        P("cfar_train_range", "Training cells, range", "int", lo=1, hi=64),
        P("cfar_train_doppler", "Training cells, Doppler", "int", lo=1, hi=64),
    ], enable="cfar"),
]


class _LogPfaProxy:
    """Lets the form edit ``cfar_pfa`` as log10 while other fields pass through."""

    def __init__(self, proc):
        object.__setattr__(self, "_p", proc)

    def __getattr__(self, name):
        v = getattr(self._p, name)
        return float(np.log10(v)) if name == "cfar_pfa" else v

    def __setattr__(self, name, value):
        setattr(self._p, name, 10.0 ** value if name == "cfar_pfa" else value)


def _scrolled(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(w)
    return area


# ----------------------------------------------------------------- main window
class MainWindow(QtWidgets.QMainWindow):
    TARGET_COLS = ["Range (m)", "Velocity (m/s)", "RCS (dBsm)", "Swerling", "Exp. SNR (dB)"]

    def __init__(self, scenario: Scenario | None = None):
        super().__init__()
        self.setWindowTitle("FMCW Radar Simulator")
        self.resize(1500, 900)
        self.rng = np.random.default_rng()
        self.scenario = scenario or next(iter(PRESETS.values()))()
        self._initial_targets = copy.deepcopy(self.scenario.targets)
        self.scene_time = 0.0
        self.last_iq: np.ndarray | None = None
        self._table_loading = False

        self._rerun = QtCore.QTimer(self, singleShot=True, interval=120)
        self._rerun.timeout.connect(self.run_frame)
        self._play = QtCore.QTimer(self, singleShot=True)
        self._play.timeout.connect(self._advance)

        self._build_toolbar()
        self._build_controls()
        self._build_plots()
        self._load_scenario_into_ui()
        self.run_frame()

    # ------------------------------------------------------------- UI building
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" Preset: "))
        self.preset_box = QtWidgets.QComboBox()
        self.preset_box.addItems(list(PRESETS))
        self.preset_box.activated.connect(lambda i: self.set_scenario(PRESETS[self.preset_box.itemText(i)]()))
        tb.addWidget(self.preset_box)
        tb.addSeparator()
        self.act_play = tb.addAction("Play")
        self.act_play.setCheckable(True)
        self.act_play.toggled.connect(self._toggle_play)
        tb.addAction("Step", self._advance_once)
        tb.addAction("Reset scene", self.reset_scene)
        tb.addWidget(QtWidgets.QLabel("  Scene step: "))
        self.step_box = QtWidgets.QDoubleSpinBox()
        self.step_box.setRange(0.0, 60.0)
        self.step_box.setDecimals(2)
        self.step_box.setSingleStep(0.1)
        self.step_box.setValue(0.5)
        self.step_box.setSuffix(" s")
        self.step_box.setToolTip("Scene time advanced per displayed frame (at least one frame duration).")
        tb.addWidget(self.step_box)
        tb.addSeparator()
        tb.addAction("Load scenario…", self.load_scenario)
        tb.addAction("Save scenario…", self.save_scenario)
        tb.addAction("Export I/Q…", self.export_iq)

    def _build_controls(self) -> None:
        sc = self.scenario
        self.radar_form = ParamForm(sc.radar, RADAR_SPEC)
        self.imp_form = ParamForm(sc.impairments, IMPAIRMENT_SPEC)
        self.proc_form = ParamForm(_LogPfaProxy(sc.processing), PROCESSING_SPEC)
        for f in (self.radar_form, self.imp_form, self.proc_form):
            f.changed.connect(self._params_changed)

        # targets tab
        self.target_table = QtWidgets.QTableWidget(0, len(self.TARGET_COLS))
        self.target_table.setHorizontalHeaderLabels(self.TARGET_COLS)
        self.target_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.itemChanged.connect(self._target_edited)
        add = QtWidgets.QPushButton("Add target")
        rem = QtWidgets.QPushButton("Remove selected")
        add.clicked.connect(self.add_target)
        rem.clicked.connect(self.remove_target)
        tgt = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tgt)
        lay.addWidget(self.target_table)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(add)
        row.addWidget(rem)
        lay.addLayout(row)
        note = QtWidgets.QLabel("Positive velocity = receding. Expected SNR uses the mean RCS.")
        note.setWordWrap(True)
        lay.addWidget(note)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(_scrolled(self.radar_form), "Radar")
        tabs.addTab(tgt, "Targets")
        tabs.addTab(_scrolled(self.imp_form), "Impairments")
        tabs.addTab(_scrolled(self.proc_form), "Processing")

        self.derived = QtWidgets.QLabel()
        self.derived.setTextFormat(QtCore.Qt.RichText)
        self.derived.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        dbox = QtWidgets.QGroupBox("Derived parameters")
        QtWidgets.QVBoxLayout(dbox).addWidget(self.derived)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(tabs, 1)
        ll.addWidget(dbox)
        left.setMinimumWidth(400)
        left.setMaximumWidth(480)
        self._left = left

    def _build_plots(self) -> None:
        self.glw = pg.GraphicsLayoutWidget()
        self.cmap = pg.colormap.get("viridis")

        self.p_rd = self.glw.addPlot(row=0, col=0, title="Range-Doppler map (dBm)")
        self.p_rd.setLabel("bottom", "Range", units="m")
        self.p_rd.setLabel("left", "Radial velocity (+ receding)", units="m/s")
        self.img_rd = pg.ImageItem()
        self.p_rd.addItem(self.img_rd)
        self.bar = pg.ColorBarItem(colorMap=self.cmap, interactive=False, width=14)
        self.bar.setImageItem(self.img_rd, insert_in=self.p_rd)
        self.det_scatter = pg.ScatterPlotItem(
            size=14, symbol="o", pen=pg.mkPen(CARDINAL, width=2), brush=None)
        self.p_rd.addItem(self.det_scatter)

        self.p_prof = self.glw.addPlot(row=0, col=1, title="Range profile (peak over Doppler)")
        self.p_prof.setLabel("bottom", "Range", units="m")
        self.p_prof.setLabel("left", "Power (dBm)")
        self.p_prof.showGrid(x=True, y=True, alpha=0.25)
        self.c_prof = self.p_prof.plot(pen=pg.mkPen(CARDINAL, width=1))
        self.floor_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(DARK_GOLD, width=1, style=QtCore.Qt.DashLine),
            label="thermal floor", labelOpts={"position": 0.9, "color": DARK_GOLD})
        self.p_prof.addItem(self.floor_line)
        self.det_prof = pg.ScatterPlotItem(
            size=10, symbol="t", pen=pg.mkPen(CARDINAL), brush=pg.mkBrush(GOLD))
        self.p_prof.addItem(self.det_prof)

        self.p_beat = self.glw.addPlot(row=1, col=0, title="Beat signal, first chirp")
        self.p_beat.setLabel("bottom", "Time", units="s")
        self.p_beat.setLabel("left", "Amplitude (√W)")
        self.p_beat.showGrid(x=True, y=True, alpha=0.25)
        self.p_beat.addLegend(offset=(-10, 10))
        self.c_i = self.p_beat.plot(pen=pg.mkPen(CARDINAL, width=1), name="I")
        self.c_q = self.p_beat.plot(pen=pg.mkPen(DARK_GOLD, width=1), name="Q")

        self.p_spec = self.glw.addPlot(row=1, col=1, title="Beat spectrogram, first chirp (dBm)")
        self.p_spec.setLabel("bottom", "Time", units="s")
        self.p_spec.setLabel("left", "Beat frequency", units="Hz")
        self.img_spec = pg.ImageItem()
        self.img_spec.setColorMap(self.cmap)
        self.p_spec.addItem(self.img_spec)

        self.det_table = QtWidgets.QTableWidget(0, 4)
        self.det_table.setHorizontalHeaderLabels(
            ["Range (m)", "Velocity (m/s)", "Power (dBm)", "SNR vs local CFAR noise (dB)"])
        self.det_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.det_table.verticalHeader().setVisible(False)
        self.det_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.det_table.setMaximumHeight(170)
        dbox = QtWidgets.QGroupBox("CFAR detections (strongest first)")
        QtWidgets.QVBoxLayout(dbox).addWidget(self.det_table)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.glw, 1)
        rl.addWidget(dbox)

        split = QtWidgets.QSplitter()
        split.addWidget(self._left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

    # ------------------------------------------------------- scenario handling
    def set_scenario(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._initial_targets = copy.deepcopy(scenario.targets)
        self.scene_time = 0.0
        self._load_scenario_into_ui()
        for p in (self.p_rd, self.p_prof, self.p_beat, self.p_spec):
            p.enableAutoRange()
        self.run_frame()

    def _load_scenario_into_ui(self) -> None:
        sc = self.scenario
        match = [i for i, k in enumerate(PRESETS) if sc.name.startswith(k)]
        self.preset_box.setCurrentIndex(match[0] if match else -1)
        self.radar_form.set_object(sc.radar)
        self.imp_form.set_object(sc.impairments)
        self.proc_form.set_object(_LogPfaProxy(sc.processing))
        self._fill_target_table()

    def _fill_target_table(self) -> None:
        self._table_loading = True
        try:
            tbl = self.target_table
            tbl.setRowCount(len(self.scenario.targets))
            for row, t in enumerate(self.scenario.targets):
                for col, val in enumerate((t.range_m, t.velocity_mps, t.rcs_dbsm)):
                    tbl.setItem(row, col, QtWidgets.QTableWidgetItem(f"{val:.2f}"))
                combo = QtWidgets.QComboBox()
                combo.addItems([str(i) for i in range(5)])
                combo.setCurrentIndex(int(t.swerling))
                combo.currentIndexChanged.connect(lambda i, t=t: self._set_swerling(t, i))
                tbl.setCellWidget(row, 3, combo)
                snr = QtWidgets.QTableWidgetItem("")
                snr.setFlags(snr.flags() & ~QtCore.Qt.ItemIsEditable)
                tbl.setItem(row, 4, snr)
        finally:
            self._table_loading = False
        self._update_expected_snr()

    def _update_expected_snr(self, ranges_too: bool = False) -> None:
        self._table_loading = True
        try:
            for row, t in enumerate(self.scenario.targets):
                if row >= self.target_table.rowCount():
                    break
                snr = expected_snr_db(self.scenario.radar, t, self.scenario.processing)
                self.target_table.item(row, 4).setText(f"{snr:.1f}")
                if ranges_too:
                    self.target_table.item(row, 0).setText(f"{t.range_m:.2f}")
        finally:
            self._table_loading = False

    def _set_swerling(self, target: Target, model: int) -> None:
        target.swerling = int(model)
        self._snapshot_targets()
        self._rerun.start()

    def _target_edited(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._table_loading or item.column() > 2:
            return
        t = self.scenario.targets[item.row()]
        attr = ("range_m", "velocity_mps", "rcs_dbsm")[item.column()]
        try:
            value = float(item.text())
        except ValueError:
            value = getattr(t, attr)
        if attr == "range_m":
            value = max(value, 0.01)
        setattr(t, attr, value)
        self._table_loading = True
        item.setText(f"{value:.2f}")
        self._table_loading = False
        self._snapshot_targets()
        self._rerun.start()

    def _snapshot_targets(self) -> None:
        """Edits define the state that 'Reset scene' returns to."""
        self._initial_targets = copy.deepcopy(self.scenario.targets)
        self.scene_time = 0.0

    def add_target(self) -> None:
        r = self.scenario.radar
        self.scenario.targets.append(Target(round(r.max_range_m * 0.4, 1), 0.0, 10.0, 0))
        self._snapshot_targets()
        self._fill_target_table()
        self._rerun.start()

    def remove_target(self) -> None:
        rows = sorted({i.row() for i in self.target_table.selectedIndexes()}, reverse=True)
        for row in rows:
            del self.scenario.targets[row]
        if rows:
            self._snapshot_targets()
            self._fill_target_table()
            self._rerun.start()

    def reset_scene(self) -> None:
        self.scenario.targets = copy.deepcopy(self._initial_targets)
        self.scene_time = 0.0
        self._fill_target_table()
        self.run_frame()

    # ------------------------------------------------------------------ files
    def load_scenario(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load scenario", "", "Scenario (*.json)")
        if not path:
            return
        try:
            self.set_scenario(io.load_scenario(path))
        except Exception as exc:  # malformed file
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))

    def save_scenario(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save scenario", "scenario.json", "Scenario (*.json)")
        if path:
            io.save_scenario(self.scenario, path)
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def export_iq(self) -> None:
        if self.last_iq is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export I/Q frame", "frame.npz", "NumPy archive (*.npz)")
        if path:
            io.export_iq(path, self.last_iq, self.scenario)
            self.statusBar().showMessage(f"Exported {path}", 4000)

    # -------------------------------------------------------------- animation
    def _toggle_play(self, on: bool) -> None:
        self.act_play.setText("Pause" if on else "Play")
        if on:
            self._play.start(0)
        else:
            self._play.stop()

    def _advance_once(self) -> None:
        dt = max(self.step_box.value(), self.scenario.radar.frame_time_s)
        for t in self.scenario.targets:
            t.range_m = max(t.range_m + t.velocity_mps * dt, 0.01)
        self.scene_time += dt
        self.run_frame()
        self._update_expected_snr(ranges_too=True)

    def _advance(self) -> None:
        self._advance_once()
        if self.act_play.isChecked():
            self._play.start(30)

    # --------------------------------------------------------------- simulate
    def _params_changed(self) -> None:
        self._rerun.start()

    def run_frame(self) -> None:
        sc = self.scenario
        r = sc.radar
        problems = r.problems()
        self._update_derived()
        if problems:
            self.statusBar().showMessage("Not simulated: " + " ".join(problems))
            return
        t_start = time.perf_counter()
        iq = simulate_frame(r, sc.targets, sc.impairments, self.rng, self.scene_time)
        rd = range_doppler(iq, r, sc.processing)
        self.last_iq = iq
        self._draw(iq, rd)
        self._update_expected_snr()
        elapsed = (time.perf_counter() - t_start) * 1e3

        notes = []
        for i, t in enumerate(sc.targets, 1):
            if r.beat_frequency_hz(t.range_m, t.velocity_mps) > r.max_beat_hz:
                notes.append(f"target {i} is beyond the unambiguous range")
            if abs(t.velocity_mps) > r.max_velocity_mps:
                notes.append(f"target {i} aliases in velocity")
        msg = f"t = {self.scene_time:.2f} s   |   frame {iq.shape[0]} × {iq.shape[1]} samples in {elapsed:.0f} ms"
        if notes:
            msg += "   |   Note: " + "; ".join(notes)
        self.statusBar().showMessage(msg)

    def _draw(self, iq: np.ndarray, rd) -> None:
        r = self.scenario.radar
        floor_dbm = 10.0 * np.log10(rd.noise_floor_w) + 30.0
        img = rd.power_dbm
        lo = floor_dbm - 5.0
        hi = max(float(img.max()), lo + 30.0)
        dr = rd.range_m[1] - rd.range_m[0]
        dv = rd.velocity_mps[1] - rd.velocity_mps[0]
        self.img_rd.setImage(img, autoLevels=False)
        self.img_rd.setRect(QtCore.QRectF(
            rd.range_m[0] - dr / 2, rd.velocity_mps[0] - dv / 2, dr * img.shape[1], dv * img.shape[0]))
        self.bar.setLevels((lo, hi))

        self.det_scatter.setData([d.range_m for d in rd.detections], [d.velocity_mps for d in rd.detections])
        self.c_prof.setData(rd.range_m, range_profile_dbm(rd))
        self.floor_line.setValue(floor_dbm)
        self.det_prof.setData([d.range_m for d in rd.detections], [d.power_dbm + 2.0 for d in rd.detections])

        t = np.arange(iq.shape[1]) / r.fs_hz
        self.c_i.setData(t, iq[0].real)
        self.c_q.setData(t, iq[0].imag)

        f, ts, sxx = beat_spectrogram(iq[0], r.fs_hz)
        self.img_spec.setImage(sxx, autoLevels=False)
        top = float(sxx.max())
        self.img_spec.setLevels((max(top - 70.0, float(np.median(sxx)) - 10.0), top))
        df = f[1] - f[0]
        dts = ts[1] - ts[0] if len(ts) > 1 else t[-1]
        self.img_spec.setRect(QtCore.QRectF(ts[0] - dts / 2, f[0] - df / 2, dts * sxx.shape[1], df * sxx.shape[0]))

        self.det_table.setRowCount(len(rd.detections))
        for row, d in enumerate(rd.detections):
            for col, val in enumerate((d.range_m, d.velocity_mps, d.power_dbm, d.snr_db)):
                self.det_table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{val:.2f}"))

    def _update_derived(self) -> None:
        r = self.scenario.radar
        rows = [
            ("Wavelength", f"{r.wavelength_m * 100:.3f} cm"),
            ("Chirp slope", f"{r.slope_hz_per_s / 1e12:.4g} MHz/µs"),
            ("Samples per chirp", f"{r.n_samples}"),
            ("Frame time", f"{r.frame_time_s * 1e3:.2f} ms"),
            ("Range resolution", f"{r.range_resolution_m:.3f} m"),
            ("Max unambiguous range", f"{r.max_range_m:,.0f} m"),
            ("Max beat frequency", f"{r.max_beat_hz / 1e6:.3f} MHz"),
            ("Velocity resolution", f"{r.velocity_resolution_mps:.3f} m/s"),
            ("Max unambiguous velocity", f"±{r.max_velocity_mps:.2f} m/s"),
            ("Thermal noise (kT₀F·fs)", f"{10 * np.log10(r.noise_power_w) + 30:.1f} dBm"),
            ("Processing gain (N·M)", f"{10 * np.log10(r.n_samples * r.n_chirps):.1f} dB"),
        ]
        self.derived.setText(
            "<table cellspacing='3'>"
            + "".join(f"<tr><td>{k}&nbsp;&nbsp;</td><td><b>{v}</b></td></tr>" for k, v in rows)
            + "</table>")


def run(scenario: Scenario | None = None) -> int:
    pg.setConfigOptions(background="w", foreground="k", imageAxisOrder="row-major", antialias=False)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("FMCW Radar Simulator")
    win = MainWindow(scenario)
    win.show()
    return app.exec()
