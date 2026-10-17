"""Windows between the beam and the detector: an editable, ordered list"""

from typing import List

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QHeaderView, QPushButton, QSizePolicy, QTableWidget, QVBoxLayout, QWidget,
)

from core.beamline import UNIT_LABELS, Window
from core.materials import MATERIAL_DB, WINDOW_MATERIAL_DB, get_material

_DEFAULT_MATERIAL = "Mylar (C10H8O4)"


def _material_names() -> list[str]:
    """Window materials first, then the detector media (a gas cell or a solid can be a layer too)"""
    names = list(WINDOW_MATERIAL_DB)
    names += [n for n in MATERIAL_DB if n not in WINDOW_MATERIAL_DB]
    return names


class WindowsPanel(QWidget):
    """
    Ordered list of windows (first row = first window the beam meets). The group box
    is checkable: unchecked means no windows, without losing the list.
    """

    valueChanged = Signal()

    def __init__(self):
        super().__init__()
        self.box = QGroupBox("Windows (beam → detector)")
        self.box.setCheckable(True)
        self.box.setChecked(False)
        self.box.setToolTip("With windows on, E₀ is the beam energy before the first window.\n"
                            "First row = first window the beam meets.")

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Material", "Thickness", "Unit"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(28)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)            # names get the spare width
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 72)
        self.table.setColumnWidth(2, 76)
        self.table.setMinimumHeight(150)

        self.btn_add = QPushButton("+ Add")
        self.btn_remove = QPushButton("− Remove")
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        for b, tip in ((self.btn_up, "Move up (earlier in the beam)"),
                       (self.btn_down, "Move down (closer to the detector)")):
            b.setFixedWidth(32)
            b.setToolTip(tip)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_up)
        buttons.addWidget(self.btn_down)

        inner = QVBoxLayout(self.box)
        inner.addWidget(self.table)
        inner.addLayout(buttons)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.box)

        self.btn_add.clicked.connect(lambda: self.add_window())
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_up.clicked.connect(lambda: self._move_selected(-1))
        self.btn_down.clicked.connect(lambda: self._move_selected(+1))
        self.box.toggled.connect(lambda _: self.valueChanged.emit())

    # the group box is the switch; keep the old QGroupBox-style API for callers
    def isChecked(self) -> bool:
        return self.box.isChecked()

    def setChecked(self, on: bool):
        self.box.setChecked(on)

    # ---- rows ----

    def add_window(self, material: str = _DEFAULT_MATERIAL, thickness: float = 1.0,
                   unit: str = "um", row: int | None = None):
        names = _material_names()
        if material not in names:
            material = _DEFAULT_MATERIAL if _DEFAULT_MATERIAL in names else names[0]
        row = self.table.rowCount() if row is None else row
        self.table.insertRow(row)

        mat = QComboBox()
        mat.addItems(names)
        mat.setCurrentText(material)
        mat.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)      # fill the cell, never widen it
        mat.view().setMinimumWidth(mat.view().sizeHintForColumn(0) + 24)  # the open list shows whole names
        mat.setToolTip(material)

        thick = QDoubleSpinBox()
        thick.setRange(0.0, 1e6)
        thick.setDecimals(3)
        thick.setButtonSymbols(QAbstractSpinBox.NoButtons)
        thick.setValue(thickness)

        units = QComboBox()
        for key, label in UNIT_LABELS.items():
            units.addItem(label, key)
        units.setCurrentIndex(max(0, units.findData(unit)))

        self.table.setCellWidget(row, 0, mat)
        self.table.setCellWidget(row, 1, thick)
        self.table.setCellWidget(row, 2, units)

        def on_material(name, combo=mat):
            combo.setToolTip(name)
            self.valueChanged.emit()

        mat.currentTextChanged.connect(on_material)
        thick.valueChanged.connect(lambda _: self.valueChanged.emit())
        units.currentIndexChanged.connect(lambda _: self.valueChanged.emit())
        self.table.selectRow(row)
        self.valueChanged.emit()

    def _row_state(self, row: int) -> tuple[str, float, str]:
        return (self.table.cellWidget(row, 0).currentText(),
                self.table.cellWidget(row, 1).value(),
                self.table.cellWidget(row, 2).currentData())

    def _remove_selected(self):
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)
            self.valueChanged.emit()

    def _move_selected(self, step: int):
        row = self.table.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        state = self._row_state(row)
        self.blockSignals(True)
        try:
            self.table.removeRow(row)
            self.add_window(*state, row=target)
        finally:
            self.blockSignals(False)
        self.table.selectRow(target)
        self.valueChanged.emit()

    # ---- model ----

    def windows(self) -> List[Window]:
        """The windows to apply (empty when the box is unchecked)"""
        if not self.isChecked():
            return []
        return [Window(get_material(name), thickness, unit)
                for name, thickness, unit in self.state()]

    def state(self) -> list[tuple[str, float, str]]:
        return [self._row_state(r) for r in range(self.table.rowCount())]

    def set_state(self, rows: list[tuple[str, float, str]], enabled: bool):
        self.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for name, thickness, unit in rows:
                self.add_window(name, thickness, unit)
            self.setChecked(enabled)
        finally:
            self.blockSignals(False)
        self.valueChanged.emit()

    # ---- session ----

    def save(self, s: QSettings):
        s.remove("windows")
        s.setValue("windows/enabled", self.isChecked())
        rows = self.state()
        s.setValue("windows/count", len(rows))
        for i, (name, thickness, unit) in enumerate(rows):
            s.setValue(f"windows/{i}/material", name)
            s.setValue(f"windows/{i}/thickness", thickness)
            s.setValue(f"windows/{i}/unit", unit)

    def load(self, s: QSettings):
        n = s.value("windows/count", 0, type=int) or 0
        rows = []
        for i in range(n):
            name = s.value(f"windows/{i}/material")
            thickness = s.value(f"windows/{i}/thickness", type=float)
            unit = s.value(f"windows/{i}/unit") or "um"
            if name and thickness is not None:
                rows.append((name, thickness, unit))
        self.blockSignals(True)
        try:
            self.set_state(rows, s.value("windows/enabled", False, type=bool))
        finally:
            self.blockSignals(False)
