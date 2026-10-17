"""Windows between the beam and the detector: an editable, ordered list"""

from copy import deepcopy
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QVBoxLayout,
)

from core.beamline import UNIT_LABELS, Window
from core.materials import MATERIAL_DB, get_material

_DEFAULT_MATERIAL = "Mylar (C10H8O4)"


class WindowsPanel(QGroupBox):
    """
    Ordered list of windows (first row = first window the beam meets). The group box
    is checkable: unchecked means no windows, without losing the list.
    """

    valueChanged = Signal()

    def __init__(self):
        super().__init__("Windows (beam → detector)")
        self.setCheckable(True)
        self.setChecked(False)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Material", "Thickness", "Unit"])
        self.table.verticalHeader().setVisible(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(110)

        self.btn_add = QPushButton("+ Add")
        self.btn_remove = QPushButton("− Remove")
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        for b in (self.btn_up, self.btn_down):
            b.setFixedWidth(32)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_up)
        buttons.addWidget(self.btn_down)

        hint = QLabel("E0 above is then the beam energy before the first window.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(hint)

        self.btn_add.clicked.connect(lambda: self.add_window())
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_up.clicked.connect(lambda: self._move_selected(-1))
        self.btn_down.clicked.connect(lambda: self._move_selected(+1))
        self.toggled.connect(lambda _: self.valueChanged.emit())

    # ---- rows ----

    def add_window(self, material: str = _DEFAULT_MATERIAL, thickness: float = 1.0,
                   unit: str = "um", row: int | None = None):
        if material not in MATERIAL_DB:
            material = _DEFAULT_MATERIAL if _DEFAULT_MATERIAL in MATERIAL_DB else next(iter(MATERIAL_DB))
        row = self.table.rowCount() if row is None else row
        self.table.insertRow(row)

        mat = QComboBox()
        mat.addItems(MATERIAL_DB.keys())
        mat.setCurrentText(material)

        thick = QDoubleSpinBox()
        thick.setRange(0.0, 1e6)
        thick.setDecimals(3)
        thick.setSingleStep(0.1)
        thick.setValue(thickness)

        units = QComboBox()
        for key, label in UNIT_LABELS.items():
            units.addItem(label, key)
        units.setCurrentIndex(max(0, units.findData(unit)))

        self.table.setCellWidget(row, 0, mat)
        self.table.setCellWidget(row, 1, thick)
        self.table.setCellWidget(row, 2, units)

        mat.currentTextChanged.connect(lambda _: self.valueChanged.emit())
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
        out = []
        for row in range(self.table.rowCount()):
            name, thickness, unit = self._row_state(row)
            out.append(Window(deepcopy(get_material(name)), thickness, unit))
        return out

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
