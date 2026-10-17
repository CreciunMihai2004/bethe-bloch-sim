"""PyQtGraph-based Bragg curve plot widget"""

from typing import List

import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore

import numpy as np

from PySide6.QtWidgets import QFileDialog

from core.detector import AnodeLayout
from core.simulation import TrackResult, find_intersections
from core.units import StoppingUnit, PlotMode

# ---- theme definitions ----
_THEMES = {
    "dark": dict(
        bg="#1e1e1e",
        fg="#ffffff",
        grid_alpha=0.3,
        cross_pen=pg.mkPen("#888888", style=QtCore.Qt.DashLine, width=1),
        label_fill=(40, 40, 40, 180),
        anode_brushes=((90, 140, 255, 28), (255, 170, 60, 22)),
        anode_pen=pg.mkPen("#9aa4b5", style=QtCore.Qt.DashLine, width=1),
    ),
    "light": dict(
        bg="#f5f5f5",
        fg="#000000",
        grid_alpha=0.25,
        cross_pen=pg.mkPen("#666666", style=QtCore.Qt.DashLine, width=1),
        label_fill=(230, 230, 230, 200),
        anode_brushes=((60, 110, 230, 30), (240, 140, 20, 26)),
        anode_pen=pg.mkPen("#6b7280", style=QtCore.Qt.DashLine, width=1),
    ),
}

class BraggPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._results: List[TrackResult] = []
        self._theme_name = "dark"
        self._unit: StoppingUnit = StoppingUnit.LINEAR_MM
        self._mass_thickness: bool = False
        self._mode        = PlotMode.DEDX
        self._anodes: AnodeLayout | None = None

        # ---- crosshair ----
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        self.addItem(self._vline, ignoreBounds=True)
        self.addItem(self._hline, ignoreBounds=True)

        self._coord_label = pg.TextItem(
            text="", anchor=(0.0, 1.0), fill=pg.mkBrush(40, 40, 40, 180)
        )
        self._coord_label.setZValue(20)
        self.addItem(self._coord_label, ignoreBounds=True)

        self._legend = self.addLegend(offset=(-10, 10))

        self._proxy = pg.SignalProxy(
            self.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_move
        )

        self._apply_theme()

    # ---- theme ----

    def _apply_theme(self):
        t = _THEMES[self._theme_name]
        self.setBackground(t["bg"])
        self.showGrid(x=True, y=True, alpha=t["grid_alpha"])

        self._vline.setPen(t["cross_pen"])
        self._hline.setPen(t["cross_pen"])
        self._coord_label.setColor(t["fg"])
        self._coord_label.fill = pg.mkBrush(*t["label_fill"])

        for axis in ("bottom", "left"):
            ax = self.getAxis(axis)
            ax.setTextPen(t["fg"])
            ax.setPen(t["fg"])
            ax.enableAutoSIPrefix(False)

    def toggle_theme(self) -> str:
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self._apply_theme()
        if self._results:
            self.redraw(self._unit, self._mass_thickness, mode=self._mode)
        return self._theme_name

    @property
    def is_dark(self) -> bool:
        return self._theme_name == "dark"

    # ---= crosshair / hover ----

    def _on_mouse_move(self, evt):
        pos = evt[0]
        if not self.sceneBoundingRect().contains(pos):
            return
        mp = self.plotItem.vb.mapSceneToView(pos)
        x, y = mp.x(), mp.y()

        self._vline.setPos(x)
        self._hline.setPos(y)

        x_unit = "g/cm²" if self._mass_thickness else "mm"

        if self._mode is PlotMode.DEDX:
            y_unit = self._unit.label
        else:
            y_unit = "MeV"

        # significant digits, not decimals: mass thickness in a gas is ~1e-4 g/cm²
        # and would otherwise read "0.000"
        self._coord_label.setText(f" x = {x:.5g} {x_unit}   y = {y:.5g} {y_unit} ")
        self._coord_label.setPos(x, y)

    # ---- data ----

    def set_results(self, results: List[TrackResult]):
        self._results = results

    def set_anodes(self, layout: AnodeLayout | None):
        """Anode geometry to shade behind the curves (None: no anodes)"""
        self._anodes = layout

    def _draw_anodes(self, mass_thickness: bool):
        if self._anodes is None:
            return
        t = _THEMES[self._theme_name]
        # in mass-thickness mode the bands follow the gas density of the run
        scale = (self._results[0].rho / 10.0) if (mass_thickness and self._results) else 1.0
        if mass_thickness and not self._results:
            return
        a = self._anodes
        for (lo, hi), brush, tip in (((a.start_mm, a.boundary_mm), t["anode_brushes"][0], "Anode 1 (ΔE)"),
                                     ((a.boundary_mm, a.end_mm), t["anode_brushes"][1], "Anode 2 (E_res)")):
            if hi <= lo:
                continue
            region = pg.LinearRegionItem(values=(lo * scale, hi * scale), movable=False,
                                         brush=pg.mkBrush(*brush), pen=pg.mkPen(None))
            region.setZValue(-10)
            region.setToolTip(tip)
            self.addItem(region)
        line = pg.InfiniteLine(pos=a.boundary_mm * scale, angle=90, movable=False, pen=t["anode_pen"],
                               label="ΔE │ E_res",
                               labelOpts=dict(position=0.97, color=t["fg"], anchors=[(0.5, 0), (0.5, 0)]))
        line.setZValue(-5)
        self.addItem(line)

    def redraw(self, unit: StoppingUnit, mass_thickness: bool, mode: PlotMode):
        self._unit = unit
        self._mass_thickness = mass_thickness
        self._mode = mode

        self.clear()
        self.addItem(self._vline, ignoreBounds=True)
        self.addItem(self._hline, ignoreBounds=True)
        self.addItem(self._coord_label, ignoreBounds=True)
        self._legend.clear()

        t = _THEMES[self._theme_name]

        x_label = "Mass thickness" if mass_thickness else "Distance x"
        x_units = "g/cm²" if mass_thickness else "mm"
        if mode is PlotMode.DEDX:
            y_label, y_units = "dE/dx", unit.label
        else:
            y_label, y_units = "Kinetic energy", "MeV"

        self.setLabel("bottom", x_label, units=x_units, **{"color": t["fg"]})
        self.setLabel("left", y_label, units=y_units, **{"color": t["fg"]})

        self._draw_anodes(mass_thickness)

        try:
            for r in self._results:
                x = r.x_in(mass_thickness)
                y = r.dEdx_in(unit) if mode is PlotMode.DEDX else r.E
                self.plot(
                    x, y,
                    pen=pg.mkPen(r.color, width=2),
                    name=r.name,
                    autoDownsample=True,
                    downsampleMethod="peak",
                )
        except Exception as e:
            print(f"[BraggPlot] curve plotting failed: {e}")

        if mode is PlotMode.DEDX and len(self._results) >= 2:
            try:
                pts = find_intersections(self._results[0], self._results[1],
                                         unit, mass_thickness)
                for i, (ix, iy) in enumerate(pts):
                    scatter = pg.ScatterPlotItem(
                        [ix], [iy],
                        symbol="x", size=12,
                        pen=pg.mkPen("red", width=2),
                        brush=pg.mkBrush("red"),
                        # only the first marker is named, otherwise the legend
                        # gets one "Intersection" row per crossing
                        name="Intersection" if i == 0 else None,
                    )
                    self.addItem(scatter)
            except Exception as e:
                print(f"[BraggPlot] intersection calc failed: {e}")

    # ---- PNG export ----

    def export_png(self, parent=None) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export plot as PNG",
            "bragg_curves.png", "PNG images (*.png)"
        )
        if not path:
            return False

        # the crosshair and coordinate label are hover helpers: keep them out of the file
        hover_items = (self._vline, self._hline, self._coord_label)
        for item in hover_items:
            item.setVisible(False)
        try:
            exporter = pyqtgraph.exporters.ImageExporter(self.plotItem)
            exporter.parameters()["width"] = int(self.width() * 2)
            exporter.export(path)
        finally:
            for item in hover_items:
                item.setVisible(True)
        return True
