"""Main application window"""

from typing import List

from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QScrollArea, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from core.detector import density_for_crossing
from core.export import export_csv, export_xlsx
from core.pipeline import SimJob, run_job, run_track
from core.simulation import SimSettings, TrackResult
from core.units import StoppingUnit, PlotMode
from ui.panels import BeamPanel, DetectorPanel, MaterialPanel, ParticlePanel
from ui.plot_widget import BraggPlot
from ui.results_view import ResultsView
from ui.windows_panel import WindowsPanel

_CONTROL_WIDTH = 380


def _scrolling(*widgets) -> QScrollArea:
    """A tab page: the widgets stacked at the top, scrolling if the screen is short"""
    page = QWidget()
    layout = QVBoxLayout(page)
    for w in widgets:
        layout.addWidget(w)
    layout.addStretch(1)
    area = QScrollArea()
    area.setWidget(page)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area


class MainWindow(QMainWindow):
    _SETTINGS_ORG = "BraggSim"
    _SETTINGS_APP = "BraggCurveSimulator"
    #   Windows : HKCU\Software\BraggSim\BraggCurveSimulator
    #   macOS   : ~/Library/Preferences/com.BraggSim.BraggCurveSimulator.plist
    #   Linux   : ~/.config/BraggSim/BraggCurveSimulator.ini

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bragg Curve Simulator — Bethe-Bloch")
        self.resize(1280, 760)

        self._results: List[TrackResult] = []

        # a run takes a few milliseconds; the timer only merges a burst of edits
        # (an arrow key held down, a mouse-wheel flick) into one run
        self._auto_run_timer = QTimer(self)
        self._auto_run_timer.setSingleShot(True)
        self._auto_run_timer.setInterval(40)
        self._auto_run_timer.timeout.connect(self.run)

        # ---- inputs, one tab per stage of the beam's path ----
        self.material_panel = MaterialPanel()
        self.p1_panel = ParticlePanel("Particle 1", "Hydrogen-1")
        self.p2_panel = ParticlePanel("Particle 2", "Helium-4")
        self.beam_panel = BeamPanel()
        self.windows_panel = WindowsPanel()
        self.detector_panel = DetectorPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(_scrolling(self.material_panel, self.p1_panel, self.p2_panel), "Setup")
        self.tabs.addTab(_scrolling(self.beam_panel, self.windows_panel), "Beamline")
        self.tabs.addTab(_scrolling(self.detector_panel), "Detector")
        self.tabs.setFixedWidth(_CONTROL_WIDTH)

        # ---- display options: one row above the plot ----
        self.plot_mode_combo = QComboBox()
        for m in PlotMode:
            self.plot_mode_combo.addItem(m.label, m)
        self.unit_combo = QComboBox()
        for u in StoppingUnit:
            self.unit_combo.addItem(u.label, u)
        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItem("Distance (mm)", False)
        self.xaxis_combo.addItem("Mass thickness (g/cm²)", True)

        self.btn_theme = QToolButton()
        self.btn_theme.setText("☀")
        self.btn_theme.setToolTip("Light / dark plot")

        self.btn_export = QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QToolButton.InstantPopup)

        display_row = QHBoxLayout()
        display_row.setContentsMargins(0, 0, 0, 0)
        for label, w in (("Plot", self.plot_mode_combo), ("dE/dx", self.unit_combo),
                         ("x", self.xaxis_combo)):
            display_row.addWidget(QLabel(label))
            display_row.addWidget(w)
            display_row.addSpacing(8)
        display_row.addStretch(1)
        display_row.addWidget(self.btn_export)
        display_row.addWidget(self.btn_theme)

        # ---- plot + results ----
        self.plot = BraggPlot()
        self.results = ResultsView()

        right = QVBoxLayout()
        right.addLayout(display_row)
        right.addWidget(self.plot, stretch=1)
        right.addWidget(self.results)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(self.tabs)
        root.addWidget(right_widget, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self._build_menus()

        # ---- signal connections ----
        for panel in (self.material_panel, self.p1_panel, self.p2_panel,
                      self.beam_panel, self.windows_panel):
            panel.valueChanged.connect(self.schedule_auto_run)
        self.detector_panel.layoutChanged.connect(self.on_display_changed)
        self.detector_panel.findDensityRequested.connect(self.place_crossing_on_boundary)

        self.plot_mode_combo.currentIndexChanged.connect(self.on_display_changed)
        self.unit_combo.currentIndexChanged.connect(self.on_display_changed)
        self.xaxis_combo.currentIndexChanged.connect(self.on_display_changed)
        self.btn_theme.clicked.connect(self._toggle_theme)

        self._sync_controls()
        self._set_export_enabled(False)

    def _build_menus(self):
        self.act_load_session = QAction("Load last session", self)
        self.act_load_session.setShortcut(QKeySequence("Ctrl+L"))
        self.act_export_csv = QAction("Export CSV…", self)
        self.act_export_xlsx = QAction("Export Excel…", self)
        self.act_export_png = QAction("Export plot as PNG…", self)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)

        self.act_load_session.triggered.connect(self._load_settings)
        self.act_export_csv.triggered.connect(self._on_export_csv)
        self.act_export_xlsx.triggered.connect(self._on_export_xlsx)
        self.act_export_png.triggered.connect(lambda: self.plot.export_png(parent=self))
        act_quit.triggered.connect(self.close)

        exports = (self.act_export_csv, self.act_export_xlsx, self.act_export_png)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_load_session)
        file_menu.addSeparator()
        file_menu.addActions(exports)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        export_menu = QMenu(self)
        export_menu.addActions(exports)
        self.btn_export.setMenu(export_menu)

    def _set_export_enabled(self, on: bool):
        for a in (self.act_export_csv, self.act_export_xlsx, self.act_export_png):
            a.setEnabled(on)
        self.btn_export.setEnabled(on)

    # ---- helpers ----

    def _current_unit(self) -> StoppingUnit:
        return self.unit_combo.currentData()

    def _current_mass_thickness(self) -> bool:
        return self.xaxis_combo.currentData()

    def _current_plot_mode(self) -> PlotMode:
        return self.plot_mode_combo.currentData()

    def _toggle_theme(self):
        dark = self.plot.toggle_theme() == "dark"
        self.btn_theme.setText("☀" if dark else "☽")

    # ---- simulation ----

    def schedule_auto_run(self, *_args):
        self._auto_run_timer.start()

    def _build_job(self) -> SimJob:
        """The current inputs as a job; ValueError with a short reason if they are not usable"""
        valid = [p for p in (self.p1_panel.particle(), self.p2_panel.particle())
                 if p.E0 is not None and p.E0 > 0]
        if not valid:
            raise ValueError("Enter a positive E₀ for at least one particle.")
        return SimJob(material=self.material_panel.material(), particles=valid,
                      settings=SimSettings(), windows=self.windows_panel.windows(),
                      beam_sigma=self.beam_panel.beam_sigma(),
                      straggling=self.beam_panel.straggling_on())

    def run(self):
        """Simulate the current inputs and show the result (a few milliseconds)"""
        self._auto_run_timer.stop()
        try:
            results = run_job(self._build_job())
        except ValueError as e:
            self._results = []
            self._set_export_enabled(False)
            self.plot.set_results([])
            self.redraw()
            self.results.show_error(str(e))
            return
        self.on_sim_done(results)

    def on_sim_done(self, results: List[TrackResult]):
        self._results = results
        self.plot.set_results(results)
        self.on_display_changed()
        self._set_export_enabled(bool(results))

    # ---- pressure finder ----

    def place_crossing_on_boundary(self):
        """Set the gas density so the two dE/dx curves cross on the anode boundary"""
        layout = self.detector_panel.layout_or_none()
        if layout is None:
            return
        try:
            job = self._build_job()
            if len(job.particles) < 2:
                raise ValueError("Both particles need a positive E₀.")
            rho = density_for_crossing(job.material, job.particles, layout.boundary_mm,
                                       lambda mat, part: run_track(job, part, mat))
        except ValueError as e:
            QMessageBox.warning(self, "Pressure finder", str(e))
            return
        if rho is None:
            QMessageBox.warning(self, "Pressure finder",
                                "The dE/dx curves of the two particles do not cross, so no "
                                "density puts a crossing on the anode boundary.")
            return
        self.material_panel.set_density(rho)
        self.run()

    # ---- display-only refresh ----

    def _sync_controls(self):
        mode = self._current_plot_mode()
        self.unit_combo.setEnabled(mode is PlotMode.DEDX)
        self.xaxis_combo.setEnabled(mode is not PlotMode.DE_E)

    def on_display_changed(self, *_args):
        self._sync_controls()
        layout = self.detector_panel.layout_or_none()
        self.plot.set_anodes(layout)
        self.redraw()
        if self._results:
            self.results.update_results(self._results, self._current_unit(),
                                        self._current_plot_mode(),
                                        self._current_mass_thickness(), layout)

    def redraw(self):
        self.plot.redraw(self._current_unit(), self._current_mass_thickness(),
                         self._current_plot_mode())

    # ---- session save + load ----

    def _panels(self):
        """(panel, extra save/load arguments) for every panel with session state"""
        return ((self.material_panel, ()), (self.p1_panel, ("particle1",)),
                (self.p2_panel, ("particle2",)), (self.beam_panel, ()),
                (self.windows_panel, ()), (self.detector_panel, ()))

    _COMBO_KEYS = (("unit_combo", "display/unit"), ("xaxis_combo", "display/xaxis"),
                   ("plot_mode_combo", "display/mode"))

    def _save_settings(self):
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        for panel, args in self._panels():
            panel.save(s, *args)
        for attr, key in self._COMBO_KEYS:
            s.setValue(key, getattr(self, attr).currentIndex())
        s.setValue("ui/tab", self.tabs.currentIndex())

    def _load_settings(self):
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        if not s.contains("material/name"):          # nothing saved yet (first launch)
            return

        for panel, args in self._panels():
            panel.load(s, *args)
        for attr, key in self._COMBO_KEYS:
            combo = getattr(self, attr)
            idx = s.value(key, type=int)
            if idx is not None and 0 <= idx < combo.count():
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        tab = s.value("ui/tab", type=int)
        if tab is not None and 0 <= tab < self.tabs.count():
            self.tabs.setCurrentIndex(tab)
        self.run()

    def closeEvent(self, event):
        self._auto_run_timer.stop()
        self._save_settings()
        super().closeEvent(event)

    # ---- export ----

    def _on_export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "bragg_curves.csv",
                                              "CSV files (*.csv)")
        if path:
            export_csv(self._results, path, self._current_unit(), self._current_mass_thickness())

    def _on_export_xlsx(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "bragg_curves.xlsx",
                                              "Excel workbook (*.xlsx)")
        if path:
            export_xlsx(self._results, path, self._current_unit(), self._current_mass_thickness(),
                        anodes=self.detector_panel.layout_or_none())
