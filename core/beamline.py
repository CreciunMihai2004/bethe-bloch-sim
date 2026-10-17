"""
Windows between the beam and the detector gas

A window is a thickness of one material from the material database. The beam passes
through the windows in order, losing energy in each (same stopping-power model as the
detector, mean energy loss only: no straggling yet), and the detector track starts at
the energy that comes out of the last one.

This is the first piece of the beamline chain; stripper gas, foils and the accelerator
stages can be added as more layers later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence

import numpy as np

from .materials import Material
from .particles import Particle
from .simulation import SimSettings, TrackResult, simulate

# thickness units: label -> how to get g/cm^2 from (value, density in g/cm^3)
THICKNESS_UNITS = {
    "um":     lambda t, rho: t * 1e-4 * rho,      # micrometre
    "nm":     lambda t, rho: t * 1e-7 * rho,      # nanometre
    "ug/cm2": lambda t, rho: t * 1e-6,            # areal density, the usual foil unit
    "mg/cm2": lambda t, rho: t * 1e-3,
}
UNIT_LABELS = {"um": "µm", "nm": "nm", "ug/cm2": "µg/cm²", "mg/cm2": "mg/cm²"}


@dataclass
class Window:
    material: Material
    thickness: float
    unit: str = "um"

    def __post_init__(self):
        if self.unit not in THICKNESS_UNITS:
            raise ValueError(f"unknown thickness unit '{self.unit}', use one of {list(THICKNESS_UNITS)}")
        if not np.isfinite(self.thickness) or self.thickness < 0:
            raise ValueError(f"window thickness must be >= 0 (got {self.thickness})")
        if self.material.rho <= 0:
            raise ValueError(f"'{self.material.name}': density must be > 0")

    @property
    def mass_thickness(self) -> float:
        """g/cm^2"""
        return THICKNESS_UNITS[self.unit](self.thickness, self.material.rho)

    @property
    def thickness_mm(self) -> float:
        return self.mass_thickness / self.material.rho * 10.0

    @property
    def label(self) -> str:
        return f"{self.material.name}, {self.thickness:g} {UNIT_LABELS[self.unit]}"


@dataclass
class WindowPassage:
    """What one window did to the beam"""
    label: str
    E_in: float     # MeV
    E_out: float    # MeV, 0 when the particle stops inside the window

    @property
    def stopped(self) -> bool:
        return self.E_out <= 0.0

    @property
    def loss(self) -> float:
        return self.E_in - self.E_out


def pass_windows(part: Particle, E_beam: float, windows: Sequence[Window]) -> List[WindowPassage]:
    """
    Energy before and after every window. Stops early if the particle comes to rest in one
    (that window is the last entry, with E_out = 0)
    """
    passages: List[WindowPassage] = []
    E = float(E_beam)
    for w in windows:
        r = simulate(w.material, replace(part, E0=E), SimSettings(x_stop=w.thickness_mm))
        # only "x_stop" means the particle came out the other side
        E_out = float(r.E[-1]) if r.stop_reason == "x_stop" else 0.0
        passages.append(WindowPassage(w.label, E, E_out))
        if E_out <= 0.0:
            break
        E = E_out
    return passages


def simulate_through_windows(detector: Material,
                             part: Particle,
                             windows: Sequence[Window],
                             settings: SimSettings,
                             progress_cb: Optional[Callable[[float], None]] = None) -> TrackResult:
    """
    The Bragg curve in the detector for a beam of energy part.E0 that first passes the windows.
    The result's E[0] is the detector entry energy, E_beam the energy before the windows.
    """
    if part.E0 is None or part.E0 <= 0:
        raise ValueError(f"{part.name}: initial kinetic energy must be > 0 MeV.")

    passages = pass_windows(part, part.E0, windows)
    E_det = passages[-1].E_out if passages else float(part.E0)

    if E_det <= 0.0:
        if progress_cb is not None:
            progress_cb(1.0)
        x0 = settings.x_start * detector.rho / 10.0
        return TrackResult(name=part.name, color=part.color,
                           x_mass=np.array([x0]), E=np.array([0.0]), dEdx_mass=np.array([0.0]),
                           rho=detector.rho, stop_reason="window",
                           E_beam=float(part.E0), windows=passages)

    r = simulate(detector, replace(part, E0=E_det), settings, progress_cb)
    r.E_beam = float(part.E0)
    r.windows = passages
    return r
