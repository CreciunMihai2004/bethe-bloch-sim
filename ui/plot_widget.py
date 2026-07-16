"""PyQtGraph-based Bragg curve plot widget"""

from typing import List

import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore

from PySide6.QtWidgets import QFileDialog

from core.simulation import TrackResult, find_intersections
from core.units import StoppingUnit, PlotMode

# ---- theme definitions ----
_THEMES = {
    "dark": dict(
        bg="#1e1e1e",
        fg="w",
        grid_alpha=0.3,
        cross_pen=pg.mkPen("#888888", style=QtCore.Qt.DashLine, width=1),
        label_fill=(40, 40, 40, 180),
    ),
    "light": dict(
        bg="#f5f5f5",
        fg="k",
        grid_alpha=0.25,
        cross_pen=pg.mkPen("#666666", style=QtCore.Qt.DashLine, width=1),
        label_fill=(230, 230, 230, 200),
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

    def toggle_theme(self) -> str:
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self._apply_theme()
        if self._results:
            self.redraw(self._unit, self._mass_thickness)
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
        
        self._coord_label.setText(f" x = {x:.3f} {x_unit}   y = {y:.4f} {y_unit} ")
        self._coord_label.setPos(x, y)

    # ---- data ----

    def set_results(self, results: List[TrackResult]):
        self._results = results

    def redraw(self, unit: StoppingUnit, mass_thickness: bool, mode: PlotMode):
        self._unit = unit
        self._mass_thickness = mass_thickness
        self._mode = mode

        self.clear()
        self.addItem(self._vline, ignoreBounds=True)
        self.addItem(self._hline, ignoreBounds=True)
        self.addItem(self._coord_label, ignoreBounds=True)
        self.addLegend(offset=(-10, 10))

        t = _THEMES[self._theme_name]

        for r in self._results:
            x = r.x_in(mass_thickness)
            y = r.dEdx_in(unit) if mode is PlotMode.DEDX else r.E
            
            self.plot(
                x, y,
                pen=pg.mkPen(r.color, width=2),
                name=r.name,
            )

        if mode is PlotMode.DEDX and len(self._results) >= 2:
            pts = find_intersections(
                self._results[0], self._results[1], unit, mass_thickness
            )
            for ix, iy in pts:
                scatter = pg.ScatterPlotItem(
                    [ix], [iy],
                    symbol="x", size=12,
                    pen=pg.mkPen("red", width=2),
                    brush=pg.mkBrush("red"),
                    name="Intersection",
                )
                self.addItem(scatter)

        x_label = "Mass thickness" if mass_thickness else "Distance x"
        x_units = "g/cm²" if mass_thickness else "mm"
        
        if mode is PlotMode.DEDX:
            y_label, y_units = "dE/dx", unit.label
        else:
            y_label, y_units = "Kinetic energy", "MeV"
            
        self.setLabel("bottom", x_label, units=x_units,
                      **{"color": t["fg"]})
        self.setLabel("left", y_label, units=y_units,
                      **{"color": t["fg"]})

    # ---- PNG export ----

    def export_png(self, parent=None) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            parent, "Export plot as PNG",
            "bragg_curves.png", "PNG images (*.png)"
        )
        if not path:
            return False
        exporter = pyqtgraph.exporters.ImageExporter(self.plotItem)
        exporter.parameters()["width"] = int(self.width() * 2)
        exporter.export(path)
        return True