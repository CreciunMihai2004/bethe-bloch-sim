"""Main application window"""

from copy import deepcopy
from typing import List

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from core.materials import MATERIAL_DB, get_material
from core.particles import PARTICLE_DB, Particle
from core.simulation import SimSettings, TrackResult, export_csv, export_xlsx
from core.units import StoppingUnit, PlotMode
from workers.sim_worker import SimJob, SimWorker
from ui.plot_widget import BraggPlot


class ParticlePanel(QGroupBox):
    """Controls for a particle"""

    valueChanged = Signal()

    def __init__(self, title: str, default_name: str, default_color: str):
        super().__init__(title)
        self._color = default_color

        self.combo = QComboBox()
        self.combo.addItems(PARTICLE_DB.keys())
        self.combo.setCurrentText(default_name)

        self.spin_z = QSpinBox()
        self.spin_z.setRange(1, 118)

        self.spin_M_u = QDoubleSpinBox()
        self.spin_M_u.setRange(0.1, 300.0)
        self.spin_M_u.setDecimals(3)
        self.spin_M_u.setSuffix(" u")
        self.spin_M_u.setSingleStep(0.1)

        self.input_E0 = QDoubleSpinBox()
        self.input_E0.setRange(0.0, 1e4)
        self.input_E0.setDecimals(3)
        self.input_E0.setSuffix(" MeV")
        self.input_E0.setSingleStep(0.1)

        form = QFormLayout(self)
        form.addRow("Preset:", self.combo)
        form.addRow("Charge z:", self.spin_z)
        form.addRow("Mass M (u):", self.spin_M_u)
        form.addRow("Energy E0 (MeV):", self.input_E0)

        self.combo.currentTextChanged.connect(self._load_preset)
        self.spin_z.valueChanged.connect(lambda _: self.valueChanged.emit())
        self.spin_M_u.valueChanged.connect(lambda _: self.valueChanged.emit())
        self.input_E0.valueChanged.connect(lambda _: self.valueChanged.emit())
        self._load_preset(default_name)

    def _load_preset(self, name: str):
        p = PARTICLE_DB[name]
        self.spin_z.setValue(p.z)
        self.spin_M_u.setValue(p.M_u)
        self.input_E0.setValue(p.E0 if p.E0 is not None else 0.0)
        self._color = p.color
        self.valueChanged.emit()

    def particle(self) -> Particle:
        e0 = self.input_E0.value()
        return Particle(
            name=self.combo.currentText(),
            z=self.spin_z.value(),
            M_u=self.spin_M_u.value(),
            E0=e0 if e0 > 0 else None,
            color=self._color,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bragg Curve Simulator — Bethe-Bloch")
        self.resize(1200, 700)

        self._results: List[TrackResult] = []
        self._thread: QThread | None = None
        self._worker: SimWorker | None = None
        self._rerun_requested = False
        self._current_run_shows_errors = False

        self._auto_run_timer = QTimer(self)
        self._auto_run_timer.setSingleShot(True)
        self._auto_run_timer.setInterval(350)
        self._auto_run_timer.timeout.connect(self._run_auto)

        # ---- material box ----
        self.mat_combo = QComboBox()
        self.mat_combo.addItems(MATERIAL_DB.keys())

        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(1e-8, 100.0)
        self.density_input.setDecimals(8)
        self.density_input.setSuffix(" g/cm³")
        self.density_input.setSingleStep(0.00001)
        self.density_input.setValue(get_material(self.mat_combo.currentText()).rho)

        mat_box = QGroupBox("Material")
        mat_form = QFormLayout(mat_box)
        mat_form.addRow("Medium:", self.mat_combo)
        mat_form.addRow("Density:", self.density_input)

        # ---- particles ----
        self.p1_panel = ParticlePanel("Particle 1", "Alpha", "cyan")
        self.p2_panel = ParticlePanel("Particle 2", "Proton", "orange")

        # ---- display options ----
        self.plot_mode_combo = QComboBox()
        for m in PlotMode:
            self.plot_mode_combo.addItem(m.label, m)
            
        self.unit_combo = QComboBox()
        for u in StoppingUnit:
            self.unit_combo.addItem(u.label, u)

        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItem("Distance (mm)", False)
        self.xaxis_combo.addItem("Mass thickness (g/cm²)", True)

        disp_box = QGroupBox("Display")
        disp_form = QFormLayout(disp_box)
        disp_form.addRow("Plot:",       self.plot_mode_combo)
        disp_form.addRow("dE/dx unit:", self.unit_combo)
        disp_form.addRow("x-axis:", self.xaxis_combo)

        # ---- action buttons ----
        self.btn_theme = QPushButton("☀  Light mode")
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_xlsx = QPushButton("Export Excel")
        self.btn_export_png = QPushButton("Export PNG")

        self.btn_export_csv.setEnabled(False)
        self.btn_export_xlsx.setEnabled(False)
        self.btn_export_png.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_export_csv)
        btn_row.addWidget(self.btn_export_xlsx)
        btn_row.addWidget(self.btn_export_png)

        # ---- left control column ----
        control = QVBoxLayout()
        control.addWidget(mat_box)
        control.addWidget(self.p1_panel)
        control.addWidget(self.p2_panel)
        control.addWidget(disp_box)
        control.addWidget(self.btn_theme)
        control.addLayout(btn_row)
        control.addStretch(1)

        control_widget = QWidget()
        control_widget.setLayout(control)
        control_widget.setFixedWidth(320)

        # ---- plot + results ----
        self.plot = BraggPlot()

        self.results_label = QLabel("Ready.")
        self.results_label.setWordWrap(True)
        self.results_label.setTextFormat(Qt.RichText)

        right = QVBoxLayout()
        right.addWidget(self.plot, stretch=1)
        right.addWidget(self.results_label)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(control_widget)
        root.addWidget(right_widget, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.statusBar().showMessage("Ready")

        # ---- signal connections ----
        self.mat_combo.currentTextChanged.connect(self._on_material_changed)
        self.density_input.valueChanged.connect(self.schedule_auto_run)

        self.p1_panel.valueChanged.connect(self.schedule_auto_run)
        self.p2_panel.valueChanged.connect(self.schedule_auto_run)

        self.plot_mode_combo.currentIndexChanged.connect(self.on_display_changed)
        self.unit_combo.currentIndexChanged.connect(self.on_display_changed)
        self.xaxis_combo.currentIndexChanged.connect(self.on_display_changed)

        self.btn_theme.clicked.connect(self._toggle_theme)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_xlsx.clicked.connect(self._on_export_xlsx)
        self.btn_export_png.clicked.connect(self._on_export_png)
        
        self._sync_controls()

    # ---- helpers ----

    def _current_unit(self) -> StoppingUnit:
        return self.unit_combo.currentData()

    def _current_mass_thickness(self) -> bool:
        return self.xaxis_combo.currentData()

    def _selected_material(self):
        material = deepcopy(get_material(self.mat_combo.currentText()))
        material.rho = self.density_input.value()
        return material
    
    def _current_plot_mode(self) -> PlotMode:
        return self.plot_mode_combo.currentData()

    # ---- material change ----

    def _on_material_changed(self, name: str):
        mat = get_material(name)
        self.density_input.blockSignals(True)
        self.density_input.setValue(mat.rho)
        self.density_input.blockSignals(False)
        self.schedule_auto_run()

    # ---- theme toggle ----

    def _toggle_theme(self):
        new_theme = self.plot.toggle_theme()
        if new_theme == "dark":
            self.btn_theme.setText("☀  Light mode")
        else:
            self.btn_theme.setText("🌙  Dark mode")

    # ---- simulation scheduling ----

    def schedule_auto_run(self, *_args):
        self._auto_run_timer.start()

    def _run_auto(self):
        self._start_simulation(show_warnings=False)

    def _build_job(self, show_warnings: bool) -> SimJob | None:
        p1 = self.p1_panel.particle()
        p2 = self.p2_panel.particle()

        valid = [p for p in (p1, p2) if p.E0 is not None and p.E0 > 0]
        if not valid:
            msg = "At least one particle needs a positive energy value."
            if show_warnings:
                QMessageBox.warning(self, "Invalid input", msg)
            else:
                self.statusBar().showMessage("Enter a positive E0 for at least one particle.")
            return None

        return SimJob(
            material=self._selected_material(),
            particles=valid,
            settings=SimSettings(),
        )

    def _start_simulation(self, show_warnings: bool):
        if self._thread is not None and self._thread.isRunning():
            self._rerun_requested = True
            return

        job = self._build_job(show_warnings)
        if job is None:
            return

        self._rerun_requested = False
        self._current_run_shows_errors = show_warnings
        self.btn_export_csv.setEnabled(False)
        self.btn_export_xlsx.setEnabled(False)
        self.btn_export_png.setEnabled(False)

        self._thread = QThread()
        self._worker = SimWorker(job)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_sim_done)
        self._worker.error.connect(self.on_sim_error)

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
        self.redraw()
        self._refresh_results_label()
        self.statusBar().showMessage("Done", 3000)

    def on_sim_error(self, msg: str):
        self.statusBar().showMessage("Error", 5000)
        if self._current_run_shows_errors:
            QMessageBox.critical(self, "Simulation error", msg)

    def _on_thread_finished(self):
        self._thread = None
        self._worker = None
        have = bool(self._results)
        self.btn_export_csv.setEnabled(have)
        self.btn_export_xlsx.setEnabled(have)
        self.btn_export_png.setEnabled(have)
        if self._rerun_requested:
            self._rerun_requested = False
            self._auto_run_timer.start(0)

    # ---- display-only refresh ----
    
    def _sync_controls(self):
        """Grey out controls that don't apply to the current plot mode"""
        self.unit_combo.setEnabled(
            self._current_plot_mode() is PlotMode.DEDX
        )

    def on_display_changed(self):
        if self._results:
            self.redraw()
            self._refresh_results_label()

    def redraw(self):
        self.plot.redraw(
            self._current_unit(),
            self._current_mass_thickness(),
            self._current_plot_mode(),
        )

    def _refresh_results_label(self):
        unit = self._current_unit()
        mode = self._current_plot_mode()
        lines = []
        for r in self._results:
            lines.append(
                f"<b>{r.name}</b>: range = {r.range_mm:.4f} mm "
                f"({r.range_mass:.5f} g/cm²), "
                f"peak dE/dx = {r.peak_dEdx(unit):.4f} {unit.label}"
            )
        self.results_label.setText("<br>".join(lines))

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
                    self._current_unit(), self._current_mass_thickness())
        self.statusBar().showMessage(f"Saved: {path}", 4000)

    def _on_export_png(self):
        if not self._results:
            return
        ok = self.plot.export_png(parent=self)
        if ok:
            self.statusBar().showMessage("Plot exported as PNG", 4000)
