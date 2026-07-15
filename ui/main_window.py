"""Main application window"""

from typing import List

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from copy import deepcopy

from core.materials import MATERIAL_DB, get_material
from core.particles import PARTICLE_DB, Particle
from core.simulation import SimSettings, TrackResult
from core.units import StoppingUnit
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
        self.spin_z.valueChanged.connect(self._notify_value_changed)
        self.spin_M_u.valueChanged.connect(self._notify_value_changed)
        self.input_E0.valueChanged.connect(self._notify_value_changed)
        self._load_preset(default_name)

    def _notify_value_changed(self, *_args):
        self.valueChanged.emit()

    def _load_preset(self, name: str):
        p = PARTICLE_DB[name]
        self.spin_z.setValue(p.z)
        self.spin_M_u.setValue(p.M_u)
        if p.E0 is None:
            self.input_E0.clear()
        else:
            self.input_E0.setValue(p.E0)
        self._color = p.color
        self.valueChanged.emit()

    def particle(self) -> Particle:
        e0_value = self.input_E0.value()
        try:
            e0 = float(e0_value) if e0_value else None
        except ValueError:
            e0 = None
        return Particle(
            name=self.combo.currentText(),
            z=self.spin_z.value(),
            M_u=self.spin_M_u.value(),
            E0=e0,
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

        # ---- control panel ----
        self.mat_combo = QComboBox()
        self.mat_combo.addItems(MATERIAL_DB.keys())

        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(1e-8, 100.0)
        self.density_input.setDecimals(6)
        self.density_input.setSuffix(" g/cm³")
        self.density_input.setSingleStep(0.0001)

        mat_box = QGroupBox("Material")
        mat_form = QFormLayout(mat_box)
        mat_form.addRow("Medium:", self.mat_combo)
        mat_form.addRow("Density:", self.density_input)

        self.p1_panel = ParticlePanel("Particle 1", "Alpha", "cyan")
        self.p2_panel = ParticlePanel("Particle 2", "Proton", "orange")

        # ---- display options ----
        self.unit_combo = QComboBox()
        for u in StoppingUnit:
            self.unit_combo.addItem(u.label, u) 

        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItem("Distance (mm)", False)
        self.xaxis_combo.addItem("Mass thickness (g/cm²)", True)

        disp_box = QGroupBox("Display")
        disp_form = QFormLayout(disp_box)
        disp_form.addRow("dE/dx unit:", self.unit_combo)
        disp_form.addRow("x-axis:", self.xaxis_combo)

        # ---- export ----
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setEnabled(False)

        control = QVBoxLayout()
        control.addWidget(mat_box)
        control.addWidget(self.p1_panel)
        control.addWidget(self.p2_panel)
        control.addWidget(disp_box)
        control.addWidget(self.btn_export)
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

        # ---- connections ----
        self.btn_export.clicked.connect(self.on_export)
        self.mat_combo.currentTextChanged.connect(self.schedule_auto_run)
        self.density_input.valueChanged.connect(self.schedule_auto_run)
        self.p1_panel.valueChanged.connect(self.schedule_auto_run)
        self.p2_panel.valueChanged.connect(self.schedule_auto_run)
        # unit / x-axis changes re-draw
        self.unit_combo.currentIndexChanged.connect(self.on_display_changed)
        self.xaxis_combo.currentIndexChanged.connect(self.on_display_changed)

    # helpers
    def _current_unit(self) -> StoppingUnit:
        return self.unit_combo.currentData()

    def _current_mass_thickness(self) -> bool:
        return self.xaxis_combo.currentData()

    def _selected_material(self):
        material = deepcopy(get_material(self.mat_combo.currentText()))
        material.rho = self.density_input.value()
        return material

    # run
    def schedule_auto_run(self, *args):
        self._auto_run_timer.start()

    def _run_auto(self):
        self._start_simulation(show_warnings=False)

    def _build_job(self, show_warnings: bool) -> SimJob | None:
        p1 = self.p1_panel.particle()
        p2 = self.p2_panel.particle()

        valid_particles = [p for p in (p1, p2) if p.E0 is not None and p.E0 > 0]
        if not valid_particles:
            if show_warnings:
                QMessageBox.warning(self, "Invalid input", "At least one particle needs a positive energy value.")
            else:
                self.statusBar().showMessage("Enter a positive E0 for at least one particle.")
            return None

        for p in (p1, p2):
            if p.E0 is not None and p.E0 <= 0:
                msg = f"{p.name}: energy must be > 0 when provided."
                if show_warnings:
                    QMessageBox.warning(self, "Invalid input", msg)
                return None

        return SimJob(
            material=self._selected_material(),
            particles=valid_particles,
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
        self.btn_export.setEnabled(False)

        self._thread = QThread()
        self._worker = SimWorker(job)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_sim_done)
        self._worker.error.connect(self.on_sim_error)

        # cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.on_thread_finished)

        self._thread.start()

    def on_sim_done(self, results: List[TrackResult]):
        self._results = results
        self.plot.set_results(results)
        self.redraw()

        unit = self._current_unit()
        lines = []
        for r in results:
            lines.append(
                f"<b>{r.name}</b>: range = {r.range_mm:.4f} mm "
                f"({r.range_mass:.5f} g/cm²), "
                f"peak dE/dx = {r.peak_dEdx(unit):.4f} {unit.label}"
            )
        self.results_label.setText("<br>".join(lines))

        self.statusBar().showMessage("Done", 3000)

    def on_sim_error(self, msg: str):
        self.statusBar().showMessage("Error", 5000)
        if self._current_run_shows_errors:
            QMessageBox.critical(self, "Simulation error", msg)

    def on_thread_finished(self):
        self._thread = None
        self._worker = None
        self.btn_export.setEnabled(bool(self._results))
        if self._rerun_requested:
            self._rerun_requested = False
            self._auto_run_timer.start(0)

    # display-only updates
    def on_display_changed(self):
        if self._results:
            self.redraw()
            # refresh the peak values in the chosen unit
            unit = self._current_unit()
            lines = []
            for r in self._results:
                lines.append(
                    f"<b>{r.name}</b>: range = {r.range_mm:.4f} mm "
                    f"({r.range_mass:.5f} g/cm²), "
                    f"peak dE/dx = {r.peak_dEdx(unit):.4f} {unit.label}"
                )
            self.results_label.setText("<br>".join(lines))

    def redraw(self):
        self.plot.redraw(self._current_unit(), self._current_mass_thickness())

    # export
    def on_export(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "bragg_curves.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        from core.simulation import export_csv
        export_csv(self._results, path,
                   self._current_unit(), self._current_mass_thickness())
        self.statusBar().showMessage(f"Saved: {path}", 4000)
