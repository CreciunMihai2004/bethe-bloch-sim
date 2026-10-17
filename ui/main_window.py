"""Main application window"""

from typing import List

from PySide6.QtCore import Qt, QThread, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QScrollArea, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from core.beamline import simulate_through_windows
from core.detector import density_for_crossing
from core.simulation import SimSettings, TrackResult, export_csv, export_xlsx
from core.units import StoppingUnit, PlotMode
from workers.sim_worker import SimJob, SimWorker
from ui.panels import DetectorPanel, MaterialPanel, ParticlePanel
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
        self._thread: QThread | None = None
        self._worker: SimWorker | None = None
        self._rerun_requested = False
        self._current_run_shows_errors = False

        self._auto_run_timer = QTimer(self)
        self._auto_run_timer.setSingleShot(True)
        self._auto_run_timer.setInterval(80)
        self._auto_run_timer.timeout.connect(self._run_auto)

        # ---- inputs, one tab per stage of the beam's path ----
        self.material_panel = MaterialPanel()
        self.p1_panel = ParticlePanel("Particle 1", "Hydrogen-1")
        self.p2_panel = ParticlePanel("Particle 2", "Helium-4")
        self.windows_panel = WindowsPanel()
        self.detector_panel = DetectorPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(_scrolling(self.material_panel, self.p1_panel, self.p2_panel), "Setup")
        self.tabs.addTab(_scrolling(self.windows_panel), "Beamline")
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
        self.statusBar().showMessage("Ready")

        # ---- signal connections ----
        for panel in (self.material_panel, self.p1_panel, self.p2_panel, self.windows_panel):
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
        self.act_export_png.triggered.connect(self._on_export_png)
        act_quit.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_load_session)
        file_menu.addSeparator()
        for a in (self.act_export_csv, self.act_export_xlsx, self.act_export_png):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        export_menu = QMenu(self)
        for a in (self.act_export_csv, self.act_export_xlsx, self.act_export_png):
            export_menu.addAction(a)
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

    def _selected_material(self):
        return self.material_panel.material()

    def _toggle_theme(self):
        dark = self.plot.toggle_theme() == "dark"
        self.btn_theme.setText("☀" if dark else "☽")

    # ---- simulation scheduling ----

    def schedule_auto_run(self, *_args):
        self._auto_run_timer.start()

    def _run_auto(self):
        self._start_simulation(show_warnings=False)

    def _warn(self, show: bool, title: str, msg: str):
        if show:
            QMessageBox.warning(self, title, msg)
        else:
            self.statusBar().showMessage(msg, 6000)

    def _build_job(self, show_warnings: bool) -> SimJob | None:
        valid = [p for p in (self.p1_panel.particle(), self.p2_panel.particle())
                 if p.E0 is not None and p.E0 > 0]
        if not valid:
            self._warn(show_warnings, "Invalid input", "Enter a positive E₀ for at least one particle.")
            return None
        try:
            windows = self.windows_panel.windows()
        except ValueError as e:
            self._warn(show_warnings, "Invalid window", f"Window: {e}")
            return None
        return SimJob(material=self._selected_material(), particles=valid,
                      settings=SimSettings(), windows=windows)

    def _start_simulation(self, show_warnings: bool):
        if self._thread is not None and self._thread.isRunning():
            self._rerun_requested = True
            return

        job = self._build_job(show_warnings)
        if job is None:
            return

        self._rerun_requested = False
        self._current_run_shows_errors = show_warnings
        self._set_export_enabled(False)

        self._thread = QThread()
        self._worker = SimWorker(job)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_sim_done)
        self._worker.error.connect(self.on_sim_error)
        self._worker.progress.connect(self._on_sim_progress)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    # ---- simulation callbacks ----

    def on_sim_done(self, results: List[TrackResult]):
        self._results = results
        self.plot.set_results(results)
        self.on_display_changed()
        self._set_export_enabled(bool(results))
        self.statusBar().showMessage("Done", 3000)

    def _on_sim_progress(self, percent: int):
        self.statusBar().showMessage(f"Simulating…  {percent} %")

    def on_sim_error(self, msg: str):
        # auto-runs do not pop up a dialog, so keep the reason visible in the status bar
        self.statusBar().showMessage(f"Error: {msg}", 8000)
        if self._current_run_shows_errors:
            QMessageBox.critical(self, "Simulation error", msg)

    def _on_thread_finished(self):
        self._thread = None
        self._worker = None
        self._set_export_enabled(bool(self._results))
        if self._rerun_requested:
            self._rerun_requested = False
            self._auto_run_timer.start(0)

    # ---- pressure finder ----

    def place_crossing_on_boundary(self):
        """Set the gas density so the two dE/dx curves cross on the anode boundary"""
        layout = self.detector_panel.layout_or_none()
        job = self._build_job(show_warnings=True)
        if layout is None or job is None:
            return
        if len(job.particles) < 2:
            self._warn(True, "Pressure finder", "Both particles need a positive E₀.")
            return

        def run(mat, part):
            return simulate_through_windows(mat, part, job.windows, job.settings)

        try:
            rho = density_for_crossing(job.material, job.particles, layout.boundary_mm, run)
        except Exception as e:                       # bad input reaches here as a simulation error
            self._warn(True, "Pressure finder", str(e))
            return
        if rho is None:
            self._warn(True, "Pressure finder",
                       "The dE/dx curves of the two particles do not cross, so no density "
                       "puts a crossing on the anode boundary.")
            return
        self.material_panel.set_density(rho)
        msg = f"Density set to {rho:.4g} g/cm³"
        if self.material_panel.is_gas():
            msg += f" ({self.material_panel.pressure_mbar():.3f} mbar)"
        self.statusBar().showMessage(msg + ": crossing on the anode boundary.", 8000)

    # ---- display-only refresh ----

    def _sync_controls(self):
        self.unit_combo.setEnabled(self._current_plot_mode() is PlotMode.DEDX)

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
        self.plot.redraw(
            self._current_unit(),
            self._current_mass_thickness(),
            self._current_plot_mode(),
        )

    # ---- session save + load ----

    def _save_settings(self):
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        self.material_panel.save(s)
        self.p1_panel.save(s, "particle1")
        self.p2_panel.save(s, "particle2")
        self.windows_panel.save(s)
        self.detector_panel.save(s)
        s.setValue("display/unit", self.unit_combo.currentIndex())
        s.setValue("display/xaxis", self.xaxis_combo.currentIndex())
        s.setValue("display/mode", self.plot_mode_combo.currentIndex())
        s.setValue("ui/tab", self.tabs.currentIndex())

    def _load_settings(self):
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)

        # nothing saved yet (first launch) -> nothing to load
        if not s.contains("material/name"):
            self.statusBar().showMessage("No saved session found.", 4000)
            return

        self.material_panel.load(s)
        self.p1_panel.load(s, "particle1")
        self.p2_panel.load(s, "particle2")
        self.windows_panel.load(s)
        self.detector_panel.load(s)

        for combo, key in ((self.unit_combo, "display/unit"),
                           (self.xaxis_combo, "display/xaxis"),
                           (self.plot_mode_combo, "display/mode")):
            idx = s.value(key, type=int)
            if idx is not None and 0 <= idx < combo.count():
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        tab = s.value("ui/tab", type=int)
        if tab is not None and 0 <= tab < self.tabs.count():
            self.tabs.setCurrentIndex(tab)

        self.on_display_changed()
        self.schedule_auto_run()

    def closeEvent(self, event):
        self._save_settings()
        self._auto_run_timer.stop()

        # Closing while a simulation is running would destroy a live QThread
        # ("QThread: Destroyed while thread is still running"), so stop it first
        try:
            if self._thread is not None and self._thread.isRunning():
                if self._worker is not None:
                    self._worker.cancel()
                self._thread.quit()
                self._thread.wait(3000)
        except RuntimeError:
            pass    # the thread object was already deleted: nothing to stop

        super().closeEvent(event)

    # ---- export ----

    def _on_export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "bragg_curves.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        export_csv(self._results, path,
                   self._current_unit(), self._current_mass_thickness())
        self.statusBar().showMessage(f"Saved: {path}", 4000)

    def _on_export_xlsx(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "bragg_curves.xlsx",
            "Excel workbook (*.xlsx)"
        )
        if not path:
            return
        export_xlsx(self._results, path,
                    self._current_unit(), self._current_mass_thickness(),
                    anodes=self.detector_panel.layout_or_none())
        self.statusBar().showMessage(f"Saved: {path}", 4000)

    def _on_export_png(self):
        if not self._results:
            return
        ok = self.plot.export_png(parent=self)
        if ok:
            self.statusBar().showMessage("Plot exported as PNG", 4000)
