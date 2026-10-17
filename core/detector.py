"""
Detector geometry: two anodes along the beam (dE, E_res)

    entrance ──gap──┬── anode 1 (dE) ──┬── anode 2 (E_res) ──┤
                    x = gap            x = boundary          x = end

All lengths in mm from the detector entrance (x = 0 of the track).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .materials import Material
from .particles import Particle
from .simulation import TrackResult, curve_crossings
from .straggling import energy_covariance


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
    # straggling (None when the track was computed without it): 1-sigma widths in MeV and
    # the correlation of dE and E_res, from the energy covariance along the track
    sigma_dE: Optional[float] = None
    sigma_E_res: Optional[float] = None
    corr: Optional[float] = None

    @property
    def covariance(self) -> Optional[np.ndarray]:
        """2x2 covariance of (E_res, dE) in MeV^2 -- E_res first, as on a dE-E plot"""
        if self.sigma_dE is None:
            return None
        c = self.corr * self.sigma_dE * self.sigma_E_res
        return np.array([[self.sigma_E_res ** 2, c], [c, self.sigma_dE ** 2]])


def anode_signals(track: TrackResult, layout: AnodeLayout) -> AnodeSignals:
    """Split the energy the track loses in the detector between the regions (mean energy loss)"""
    x, E = track.x_mm - track.x_mm[0], track.E

    def E_at(d: float) -> float:
        if len(x) < 2:
            return float(E[-1])
        return float(np.interp(d, x, E, right=E[-1]))

    e0, e1, e2, e3 = E[0], E_at(layout.start_mm), E_at(layout.boundary_mm), E_at(layout.end_mm)
    signals = AnodeSignals(gap=e0 - e1, dE=e1 - e2, E_res=e2 - e3, escaped=e3)

    if track.strag_J is not None and len(x) > 1:
        C = energy_covariance(track, [layout.start_mm, layout.boundary_mm, layout.end_mm])
        v_dE, v_res = np.array([1.0, -1.0, 0.0]), np.array([0.0, 1.0, -1.0])
        var_dE, var_res = v_dE @ C @ v_dE, v_res @ C @ v_res
        cov = v_dE @ C @ v_res
        signals.sigma_dE = float(np.sqrt(max(var_dE, 0.0)))
        signals.sigma_E_res = float(np.sqrt(max(var_res, 0.0)))
        denom = signals.sigma_dE * signals.sigma_E_res
        signals.corr = float(np.clip(cov / denom, -1.0, 1.0)) if denom > 0 else 0.0
    return signals


def separation(a: AnodeSignals, b: AnodeSignals) -> Optional[float]:
    """
    How far apart two particles' (E_res, dE) spots are, in units of their spread:
    the Mahalanobis distance between the centres with the average covariance.
    2 means the 1-sigma ellipses just touch; None without straggling.
    """
    Ca, Cb = a.covariance, b.covariance
    if Ca is None or Cb is None:
        return None
    d = np.array([a.E_res - b.E_res, a.dE - b.dE])
    C = 0.5 * (Ca + Cb)
    try:
        return float(np.sqrt(d @ np.linalg.solve(C, d)))
    except np.linalg.LinAlgError:
        return None


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
        mat = replace(detector, rho=rho)
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
