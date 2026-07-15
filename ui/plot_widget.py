"""PyQtGraph-based Bragg curve plot widget"""

from typing import List

import pyqtgraph as pg

from core.simulation import TrackResult, find_intersections
from core.units import StoppingUnit


class BraggPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#1e1e1e")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.addLegend(offset=(-10, 10))
        self._results: List[TrackResult] = []

    def set_results(self, results: List[TrackResult]):
        """Store new simulation results (does not draw)."""
        self._results = results

    def redraw(self, unit: StoppingUnit, mass_thickness: bool):
        self.clear()
        self.addLegend(offset=(-10, 10))

        for r in self._results:
            self.plot(
                r.x_in(mass_thickness), r.dEdx_in(unit),
                pen=pg.mkPen(r.color, width=2), name=r.name,
            )

        if len(self._results) >= 2:
            pts = find_intersections(
                self._results[0], self._results[1], unit, mass_thickness
            )
            for (ix, iy) in pts:
                scatter = pg.ScatterPlotItem(
                    [ix], [iy], symbol="x", size=12,
                    pen=pg.mkPen("red", width=2), brush=pg.mkBrush("red"),
                    name="Intersection",
                )
                self.addItem(scatter)

        x_label = "Mass thickness" if mass_thickness else "Distance x"
        x_units = "g/cm²" if mass_thickness else "mm"
        self.setLabel("bottom", x_label, units=x_units)
        self.setLabel("left", "dE/dx", units=unit.label)