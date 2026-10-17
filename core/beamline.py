"""
Windows between the beam and the detector gas

A window is a thickness of one material from the material database. The beam passes
through the windows in order, losing energy in each (same stopping-power model as the
detector), and the detector track starts at the energy that comes out of the last one.
The energy spread (beam spread + straggling in every window, see core/straggling.py) is
carried along and handed to the detector track.

This is the first piece of the beamline chain; stripper gas, foils and the accelerator
stages can be added as more layers later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import List, Sequence

import numpy as np

from .materials import Material
from .particles import Particle
from .simulation import SimSettings, TrackResult, simulate
from .straggling import track_reduced_variance

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
    E_in: float             # MeV
    E_out: float            # MeV, 0 when the particle stops inside the window
    sigma_out: float = 0.0  # MeV, 1-sigma energy spread leaving the window

    @property
    def stopped(self) -> bool:
        return self.E_out <= 0.0

    @property
    def loss(self) -> float:
        return self.E_in - self.E_out


def pass_windows(part: Particle, E_beam: float, windows: Sequence[Window],
                 sigma_beam: float = 0.0, straggling: bool = True) -> List[WindowPassage]:
    """
    Energy (and its 1-sigma spread) before and after every window. Stops early if the
    particle comes to rest in one (that window is the last entry, with E_out = 0)
    """
    passages: List[WindowPassage] = []
    E, var = float(E_beam), float(sigma_beam) ** 2
    for w in windows:
        p = replace(part, E0=E)
        r = simulate(w.material, p, SimSettings(x_stop=w.thickness_mm))
        # only "x_stop" means the particle came out the other side
        if r.stop_reason != "x_stop":
            passages.append(WindowPassage(w.label, E, 0.0, 0.0))
            break
        E_out = float(r.E[-1])
        if straggling and len(r.E) > 1:
            J = track_reduced_variance(r, w.material, p, var)
            var = float(r.dEdx_mass[-1] ** 2 * J[-1])
        passages.append(WindowPassage(w.label, E, E_out, sqrt(var)))
        E = E_out
    return passages


def simulate_through_windows(detector: Material,
                             part: Particle,
                             windows: Sequence[Window],
                             settings: SimSettings,
                             sigma_beam: float = 0.0,
                             straggling: bool = True) -> TrackResult:
    """
    The Bragg curve in the detector for a beam of energy part.E0 (1-sigma spread sigma_beam,
    MeV) that first passes the windows. The result's E[0] is the detector entry energy,
    E_beam the energy before the windows; with straggling the track carries strag_J, so
    its sigma_E, range spread and anode-signal widths are available.
    """
    if part.E0 is None or part.E0 <= 0:
        raise ValueError(f"{part.name}: initial kinetic energy must be > 0 MeV.")

    passages = pass_windows(part, part.E0, windows, sigma_beam, straggling)
    E_det = passages[-1].E_out if passages else float(part.E0)
    var_det = (passages[-1].sigma_out if passages else float(sigma_beam)) ** 2

    if E_det <= 0.0:
        x0 = settings.x_start * detector.rho / 10.0
        return TrackResult(name=part.name, color=part.color,
                           x_mass=np.array([x0]), E=np.array([0.0]), dEdx_mass=np.array([0.0]),
                           rho=detector.rho, stop_reason="window",
                           E_beam=float(part.E0), windows=passages)

    p = replace(part, E0=E_det)
    r = simulate(detector, p, settings)
    r.E_beam = float(part.E0)
    r.windows = passages
    if straggling and len(r.E) > 1:
        r.strag_J = track_reduced_variance(r, detector, p, var_det)
    return r
