"""Input panels of the control column: detector gas, particles, anodes"""

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from core.detector import AnodeLayout
from core.materials import MATERIAL_DB, Material, get_material
from core.particles import PARTICLE_DB, Particle
from ui.widgets import ScientificSpinBox

# a medium whose reference (STP) density is below this is a gas: show its pressure
_GAS_MAX_DENSITY = 0.01     # g/cm^3


def _spin(lo, hi, decimals, step, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    return s


class MaterialPanel(QGroupBox):
    """Detector medium and its density; the pressure is shown for gases"""

    valueChanged = Signal()

    def __init__(self):
        super().__init__("Detector gas")
        self.combo = QComboBox()
        self.combo.addItems(MATERIAL_DB.keys())

        self.density = ScientificSpinBox()
        self.density.setRange(1e-12, 100.0)
        self.density.setSuffix(" g/cm³")
        self.density.setToolTip("Type e.g. 1.08e-5; the arrows step by 1 %")
        self.pressure_label = QLabel()
        self.pressure_label.setToolTip("Pressure at 0 °C, scaled from the reference density in materials.csv")

        form = QFormLayout(self)
        form.addRow("Medium:", self.combo)
        form.addRow("Density:", self.density)
        form.addRow("Pressure:", self.pressure_label)
        self._form = form

        self.combo.currentTextChanged.connect(self._on_material_changed)
        self.density.valueChanged.connect(self._on_density_changed)
        self._on_material_changed(self.combo.currentText(), emit=False)

    def _reference(self) -> Material:
        return get_material(self.combo.currentText())

    def _on_material_changed(self, _name: str, emit: bool = True):
        self.density.blockSignals(True)
        self.density.setValue(self._reference().rho)
        self.density.blockSignals(False)
        self._update_pressure()
        if emit:
            self.valueChanged.emit()

    def _on_density_changed(self, _v):
        self._update_pressure()
        self.valueChanged.emit()

    def is_gas(self) -> bool:
        return self._reference().rho < _GAS_MAX_DENSITY

    def pressure_mbar(self) -> float:
        rho0 = self._reference().rho
        return 1013.25 * self.density.value() / rho0 if rho0 > 0 else float("nan")

    def _update_pressure(self):
        gas = self.is_gas()
        self._form.setRowVisible(self.pressure_label, gas)
        if gas:
            self.pressure_label.setText(f"{self.pressure_mbar():.3f} mbar")

    def material(self) -> Material:
        mat = self._reference()
        mat.rho = self.density.value()
        return mat

    def set_density(self, rho: float):
        self.density.setValue(rho)

    # ---- session ----

    def save(self, s: QSettings):
        s.setValue("material/name", self.combo.currentText())
        s.setValue("material/density", self.density.value())

    def load(self, s: QSettings):
        self.blockSignals(True)
        try:
            name = s.value("material/name")
            if name and name in MATERIAL_DB:
                self.combo.setCurrentText(name)
            rho = s.value("material/density", type=float)
            if rho and rho > 0:
                self.density.setValue(rho)
        finally:
            self.blockSignals(False)
        self._update_pressure()


class ParticlePanel(QGroupBox):
    """Controls for a particle"""

    valueChanged = Signal()

    def __init__(self, title: str, default_name: str):
        super().__init__(title)

        self.combo = QComboBox()
        self.combo.addItems(PARTICLE_DB.keys())
        self.combo.setCurrentText(default_name)

        self.spin_z = QSpinBox()
        self.spin_z.setRange(1, 118)
        self.spin_z.setPrefix("z = ")

        self.spin_M_u = _spin(0.1, 300.0, 3, 0.1, " u")

        self.input_E0 = _spin(0.0, 1e4, 3, 0.1, " MeV")

        zm = QHBoxLayout()
        zm.setContentsMargins(0, 0, 0, 0)
        zm.addWidget(self.spin_z)
        zm.addWidget(self.spin_M_u, stretch=1)
        zm_widget = QWidget()
        zm_widget.setLayout(zm)

        form = QFormLayout(self)
        form.addRow("Preset:", self.combo)
        form.addRow("Charge, mass:", zm_widget)
        form.addRow("Energy E₀:", self.input_E0)

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
        z, M_u = self.spin_z.value(), self.spin_M_u.value()

        # z or M edited away from the preset: do not label the curve with the preset's name
        preset = PARTICLE_DB.get(self.combo.currentText())
        name = self.combo.currentText()
        if preset is not None and (z != preset.z or abs(M_u - preset.M_u) > 5e-4):
            name += " (custom)"

        return Particle(
            name=name,
            z=z,
            M_u=M_u,
            E0=e0 if e0 > 0 else None,
            color=self._color,
        )

    # ---- session ----

    def save(self, s: QSettings, group: str):
        s.setValue(f"{group}/preset", self.combo.currentText())
        s.setValue(f"{group}/z", self.spin_z.value())
        s.setValue(f"{group}/M_u", self.spin_M_u.value())
        s.setValue(f"{group}/E0", self.input_E0.value())

    def load(self, s: QSettings, group: str):
        self.blockSignals(True)
        try:
            preset = s.value(f"{group}/preset")
            if preset and preset in PARTICLE_DB:
                self.combo.blockSignals(True)
                self.combo.setCurrentText(preset)
                self.combo.blockSignals(False)
                self._load_preset(preset)
            z = s.value(f"{group}/z", type=int)
            if z:
                self.spin_z.setValue(z)
            M_u = s.value(f"{group}/M_u", type=float)
            if M_u and M_u > 0:
                self.spin_M_u.setValue(M_u)
            E0 = s.value(f"{group}/E0", type=float)
            if E0 and E0 > 0:
                self.input_E0.setValue(E0)
        finally:
            self.blockSignals(False)


class DetectorPanel(QWidget):
    """
    Anode geometry and the pressure finder. Changing the geometry only changes the
    display (anode bands, dE / E_res), so it emits layoutChanged, not a re-run.
    """

    layoutChanged = Signal()
    findDensityRequested = Signal()

    def __init__(self):
        super().__init__()
        self.box = QGroupBox("Anodes")
        self.box.setCheckable(True)
        self.box.setChecked(True)

        d = AnodeLayout()
        self.gap = _spin(0.0, 1e4, 1, 1.0, " mm")
        self.gap.setValue(d.gap_mm)
        self.gap.setToolTip("Gas between the entrance window and the first anode (not read out)")
        self.anode1 = _spin(0.0, 1e4, 1, 1.0, " mm")
        self.anode1.setValue(d.anode1_mm)
        self.anode2 = _spin(0.0, 1e4, 1, 1.0, " mm")
        self.anode2.setValue(d.anode2_mm)

        form = QFormLayout(self.box)
        form.addRow("Entrance gap:", self.gap)
        form.addRow("Anode 1 (ΔE):", self.anode1)
        form.addRow("Anode 2 (E_res):", self.anode2)

        self.btn_find = QPushButton("Put crossing on anode boundary")
        self.btn_find.setToolTip(
            "Sets the gas density so that the dE/dx curves of particle 1 and 2\n"
            "cross exactly where anode 1 ends and anode 2 begins")
        form.addRow(self.btn_find)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.box)

        for w in (self.gap, self.anode1, self.anode2):
            w.valueChanged.connect(lambda _: self.layoutChanged.emit())
        self.box.toggled.connect(lambda _: self.layoutChanged.emit())
        self.btn_find.clicked.connect(self.findDensityRequested.emit)

    def layout_or_none(self) -> AnodeLayout | None:
        if not self.box.isChecked():
            return None
        return AnodeLayout(self.gap.value(), self.anode1.value(), self.anode2.value())

    # ---- session ----

    def save(self, s: QSettings):
        s.setValue("detector/anodes", self.box.isChecked())
        s.setValue("detector/gap_mm", self.gap.value())
        s.setValue("detector/anode1_mm", self.anode1.value())
        s.setValue("detector/anode2_mm", self.anode2.value())

    def load(self, s: QSettings):
        self.blockSignals(True)
        try:
            if s.contains("detector/anodes"):
                self.box.setChecked(s.value("detector/anodes", type=bool))
            for key, w in (("gap_mm", self.gap), ("anode1_mm", self.anode1), ("anode2_mm", self.anode2)):
                v = s.value(f"detector/{key}", type=float)
                if v is not None and v >= 0 and s.contains(f"detector/{key}"):
                    w.setValue(v)
        finally:
            self.blockSignals(False)


class BeamPanel(QGroupBox):
    """Beam energy spread and the straggling switch"""

    valueChanged = Signal()

    def __init__(self):
        super().__init__("Beam")
        self.sigma = _spin(0.0, 1000.0, 2, 1.0, " keV")
        self.sigma.setToolTip("1σ energy spread of the beam before the first window")
        self.straggling = QCheckBox("Energy straggling")
        self.straggling.setChecked(True)
        self.straggling.setToolTip(
            "Bohr straggling with the effective charge and the Lindhard-Scharff\n"
            "low-velocity factor, carried through the windows and the gas.\n"
            "Not yet included: charge-exchange and nuclear-collision straggling.")

        form = QFormLayout(self)
        form.addRow("Energy spread σ:", self.sigma)
        form.addRow(self.straggling)

        self.sigma.valueChanged.connect(lambda _: self.valueChanged.emit())
        self.straggling.toggled.connect(lambda _: self.valueChanged.emit())

    def beam_sigma(self) -> float:
        """MeV"""
        return self.sigma.value() * 1e-3

    def straggling_on(self) -> bool:
        return self.straggling.isChecked()

    # ---- session ----

    def save(self, s: QSettings):
        s.setValue("beam/sigma_keV", self.sigma.value())
        s.setValue("beam/straggling", self.straggling.isChecked())

    def load(self, s: QSettings):
        self.blockSignals(True)
        try:
            v = s.value("beam/sigma_keV", type=float)
            if v is not None and v >= 0 and s.contains("beam/sigma_keV"):
                self.sigma.setValue(v)
            if s.contains("beam/straggling"):
                self.straggling.setChecked(s.value("beam/straggling", type=bool))
        finally:
            self.blockSignals(False)
