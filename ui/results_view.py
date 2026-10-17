"""Compact results under the plot: one table row per particle, one line for the crossing"""

from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from core.detector import AnodeLayout, anode_signals, separation
from core.simulation import TrackResult, find_intersections
from core.units import PlotMode, StoppingUnit

# the model-limit note is only worth showing when a noticeable share of the energy is untracked
_UNSPENT_WARN_FRACTION = 0.02
_AMBER = "#e8a33d"


def track_warning(r: TrackResult, layout: Optional[AnodeLayout] = None) -> str:
    """One short warning for a track whose numbers are not the whole story ('' if none)"""
    if r.stop_reason == "window":
        return (f"Stops inside window {len(r.windows)} ({r.windows[-1].label}) "
                f"and never reaches the detector.")
    if r.stop_reason in ("x_stop", "max_steps"):
        why = "depth limit" if r.stop_reason == "x_stop" else "step limit"
        return f"Track cut by the {why} with {r.E[-1]:.3f} MeV left: the range is a lower bound."
    if r.stop_reason == "model_limit" and r.unspent_fraction > _UNSPENT_WARN_FRACTION:
        return (f"Tracking stops at {r.E[-1] * 1000:.0f} keV ({r.unspent_fraction:.0%} of the "
                f"entry energy), where the stopping model is not valid any more.")
    if layout is not None:
        s = anode_signals(r, layout)
        if s.escaped > 0:
            return (f"Leaves anode 2 with {s.escaped * 1000:.0f} keV: the residual energy "
                    f"is not fully collected.")
        if r.range_mm < layout.boundary_mm:
            return "Stops before anode 2: no residual-energy signal."
    return ""


def _pm(value: float, sigma, digits: int) -> str:
    """'value ± sigma', or just the value without straggling"""
    if sigma is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {sigma:.{digits}f}"


def windows_tooltip(r: TrackResult) -> str:
    if not r.windows:
        return ""
    lines = [f"Beam: {r.E_beam:.4g} MeV"]
    for i, w in enumerate(r.windows, 1):
        spread = f" (σ {w.sigma_out * 1000:.1f} keV)" if w.sigma_out > 0 else ""
        lines.append(f"{i}. {w.label}: −{w.loss * 1000:.1f} keV → {w.E_out:.4g} MeV{spread}")
    return "\n".join(lines)


class ResultsView(QWidget):
    COLUMNS = ["Particle", "E in (MeV)", "Range (mm)", "Peak dE/dx", "ΔE (MeV)", "E_res (MeV)", ""]
    _DE, _ERES, _WARN = 4, 5, 6

    def __init__(self):
        super().__init__()
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(self._WARN, QHeaderView.Fixed)
        self.table.setColumnWidth(self._WARN, 28)
        self.table.verticalHeader().setDefaultSectionSize(24)

        self.crossing_label = QLabel()
        self.crossing_label.setTextFormat(Qt.RichText)
        self.crossing_label.setToolTip(
            "± and the separation include the beam spread and electronic (Bohr) straggling only.\n"
            "Charge-exchange straggling, nuclear scattering and the detector resolution\n"
            "widen the real spots, so the real separation is smaller.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.table)
        layout.addWidget(self.crossing_label)
        self.show_message("Ready.")

    def _fit_height(self):
        h = self.table.horizontalHeader().height() + 4
        h += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        self.table.setFixedHeight(h)

    def show_message(self, text: str):
        self.table.setRowCount(0)
        self._fit_height()
        self.crossing_label.setText(text)

    def show_error(self, text: str):
        """A problem with the inputs or the run: replaces the results until the next good run"""
        self.show_message(f"<span style='color:{_AMBER}'>⚠ {text}</span>")

    def update_results(self, results: List[TrackResult], unit: StoppingUnit, mode: PlotMode,
                       mass_thickness: bool, layout: Optional[AnodeLayout]):
        anodes = layout is not None
        self.table.setColumnHidden(self._DE, not anodes)
        self.table.setColumnHidden(self._ERES, not anodes)
        self.table.horizontalHeaderItem(3).setToolTip(f"Peak dE/dx in {unit.label}")

        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            reached = r.stop_reason != "window"
            name = QTableWidgetItem(("● " if r.color else "") + r.name)
            name.setForeground(QColor(r.color))
            name.setToolTip(windows_tooltip(r) or "No windows")
            cells = [
                name,
                QTableWidgetItem(f"{r.E[0]:.4f}" if reached else "—"),
                QTableWidgetItem(_pm(r.range_mm, r.range_sigma_mm, 2) if reached else "—"),
                QTableWidgetItem(f"{r.peak_dEdx(unit):.5g}" if reached else "—"),
            ]
            if anodes and reached:
                s = anode_signals(r, layout)
                cells += [QTableWidgetItem(_pm(s.dE, s.sigma_dE, 4)),
                          QTableWidgetItem(_pm(s.E_res, s.sigma_E_res, 4))]
            else:
                cells += [QTableWidgetItem("—"), QTableWidgetItem("—")]
            if r.windows:
                cells[1].setToolTip(f"after {len(r.windows)} window(s), "
                                    f"−{r.window_loss * 1000:.1f} keV from {r.E_beam:.4g} MeV")
            warning = track_warning(r, layout)
            warn = QTableWidgetItem("⚠" if warning else "")
            if warning:
                warn.setForeground(QColor(_AMBER))
                warn.setToolTip(warning)
            cells.append(warn)
            for col, item in enumerate(cells):
                if col:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        self._fit_height()
        self.crossing_label.setText(
            self._crossing_text(results, unit, mode, mass_thickness, layout))

    @classmethod
    def _crossing_text(cls, results, unit, mode, mass_thickness, layout) -> str:
        if len(results) < 2:
            return ""
        sep = cls._separation_text(results, layout)
        if mode is PlotMode.DE_E:
            return sep or ("<span style='color:gray'>Switch on the anodes (Detector tab) "
                           "for the ΔE-E view.</span>" if layout is None else "")
        if mode is not PlotMode.DEDX:
            return sep
        pts = find_intersections(results[0], results[1], unit, mass_thickness)
        if not pts:
            return "<span style='color:gray'>The dE/dx curves do not cross.</span>"
        x_unit = "g/cm²" if mass_thickness else "mm"
        parts = [f"x = {x:.5g} {x_unit}, y = {y:.5g}" for x, y in pts]
        text = "<b>✕ Crossing:</b> " + " &nbsp;|&nbsp; ".join(parts) + f" {unit.label}"
        if layout is not None:
            rho = results[0].rho
            x_mm = [x / rho * 10.0 if mass_thickness else x for x, _ in pts]
            d = min(x_mm, key=lambda v: abs(v - layout.boundary_mm)) - layout.boundary_mm
            side = "past" if d > 0 else "before"
            text += (f" &nbsp;·&nbsp; {abs(d):.2f} mm {side} the anode boundary"
                     if abs(d) >= 0.005 else " &nbsp;·&nbsp; on the anode boundary")
        if sep:
            text += " &nbsp;·&nbsp; " + sep
        return text

    @staticmethod
    def _separation_text(results, layout) -> str:
        if layout is None or any(r.stop_reason == "window" for r in results[:2]):
            return ""
        d = separation(anode_signals(results[0], layout), anode_signals(results[1], layout))
        if d is None:
            return ""
        return (f"<b>ΔE-E separation:</b> {d:.1f}σ"
                if np.isfinite(d) else "<b>ΔE-E separation:</b> —")

    # ---- plain text, for tests and copying ----

    def text(self) -> str:
        lines = ["\t".join(self.COLUMNS)]
        for r in range(self.table.rowCount()):
            cells = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is None:
                    cells.append("")
                    continue
                cells.append(item.text() + (f" [{item.toolTip()}]" if item.toolTip() else ""))
            lines.append("\t".join(cells))
        lines.append(self.crossing_label.text())
        return "\n".join(lines)
