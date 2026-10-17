"""
Detector geometry: two anodes along the beam (dE, E_res)

    entrance ──gap──┬── anode 1 (dE) ──┬── anode 2 (E_res) ──┤
                    x = gap            x = boundary          x = end

All lengths in mm from the detector entrance (x = 0 of the track).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .materials import Material
from .particles import Particle
from .simulation import SimSettings, TrackResult, curve_crossings


@dataclass
class AnodeLayout:
    gap_mm: float = 0.0         # entrance window -> start of anode 1 (gas that is not read out)
    anode1_mm: float = 110.0    # dE anode (defaults)
    anode2_mm: float = 210.0    # residual-energy anode

    def __post_init__(self):
        for name in ("gap_mm", "anode1_mm", "anode2_mm"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be >= 0 (got {v})")

    @property
    def start_mm(self) -> float:
        return self.gap_mm

    @property
    def boundary_mm(self) -> float:
        """Where anode 1 ends and anode 2 begins: the target for the Bragg-curve crossing"""
        return self.gap_mm + self.anode1_mm

    @property
    def end_mm(self) -> float:
        return self.gap_mm + self.anode1_mm + self.anode2_mm


@dataclass
class AnodeSignals:
    """Energy deposited per region (MeV); their sum is the energy entering the detector"""
    gap: float
    dE: float           # anode 1
    E_res: float        # anode 2
    escaped: float      # still left at the end of anode 2 (the ion leaves the active volume)


def anode_signals(track: TrackResult, layout: AnodeLayout) -> AnodeSignals:
    """Split the energy the track loses in the detector between the regions (mean energy loss)"""
    x, E = track.x_mm - track.x_mm[0], track.E

    def E_at(d: float) -> float:
        if len(x) < 2:
            return float(E[-1])
        return float(np.interp(d, x, E, right=E[-1]))

    e0, e1, e2, e3 = E[0], E_at(layout.start_mm), E_at(layout.boundary_mm), E_at(layout.end_mm)
    return AnodeSignals(gap=e0 - e1, dE=e1 - e2, E_res=e2 - e3, escaped=e3)


def crossings_mass(a: TrackResult, b: TrackResult) -> list[float]:
    """Mass-thickness positions (g/cm^2) where the two dE/dx curves cross"""
    return [x for x, _ in curve_crossings(a.x_mass, a.dEdx_mass, b.x_mass, b.dEdx_mass)]


def density_for_crossing(detector: Material,
                         particles: Sequence[Particle],
                         x_target_mm: float,
                         run,
                         max_iter: int = 6,
                         rtol: float = 1e-6) -> Optional[float]:
    """
    Gas density (g/cm^3) that puts the crossing of the two particles' dE/dx curves at
    x_target_mm (normally the anode boundary). None if the curves do not cross.

    The mass stopping power does not depend on the density (apart from the density effect,
    which is zero at these velocities), so the crossing sits at a fixed mass thickness X and
    rho = 10 X / x_target. The loop re-runs at the new density until that holds, which also
    covers the density effect.

    run(material, particle) -> TrackResult does the simulation (with the windows), so this
    uses exactly the same physics as the plot.
    """
    if x_target_mm <= 0 or len(particles) < 2:
        return None
    rho = detector.rho
    for _ in range(max_iter):
        mat = _with_density(detector, rho)
        a, b = run(mat, particles[0]), run(mat, particles[1])
        xs = crossings_mass(a, b)
        if not xs:
            return None
        target_mass = x_target_mm * rho / 10.0
        X = min(xs, key=lambda v: abs(v - target_mass))       # the crossing nearest the target
        new_rho = 10.0 * X / x_target_mm
        if abs(new_rho - rho) <= rtol * rho:
            return new_rho
        rho = new_rho
    return rho


def _with_density(mat: Material, rho: float) -> Material:
    from dataclasses import replace
    return replace(mat, rho=rho)
