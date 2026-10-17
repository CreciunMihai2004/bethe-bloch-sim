from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .materials import Material
from .particles import Particle
from .physics import dEdx_components
from .units import StoppingUnit, convert_from_mass


@dataclass
class SimSettings:
    x_start: float = 0.0        # mm
    x_stop: Optional[float] = None   # mm; None = no depth limit (the track runs to the end)
    E_cutoff: float = 0.0001    # MeV; below this the last bit is added analytically
    dlnE: float = 0.0025        # energy-grid spacing, d(ln E) per point (0.25 % in E)


@dataclass
class TrackResult:
    """
    Bragg curve result: the track from the entry energy down to where the
    stopping-power model stops being valid (or the particle stops)

    x_mass    : mass thickness (g/cm^2)
    E         : kinetic energy (MeV)
    dEdx_mass : mass stopping power (MeV cm^2 / g)
    rho       : material density (g/cm^3)
    stop_reason : why the track ended
        "stopped"     the particle ran out of energy: E ends at exactly 0 (range reached)
        "model_limit" electronic stopping is not valid below this energy, so the track
                      stops here; the remaining slowing-down is NOT in range_mm (see
                      unspent_fraction). Only happens if Bethe-Bloch and the low-energy
                      model never cross, which is not the case for normal ions and gases.
        "x_stop"      the depth limit of SimSettings was hit first
        With "x_stop", range_mm is only a lower bound.
    """
    name: str
    color: str
    x_mass: np.ndarray
    E: np.ndarray
    dEdx_mass: np.ndarray
    rho: float
    stop_reason: str = "stopped"
    # windows in front of the detector (see core/beamline.py): the energy the beam had
    # before them and what happened in each one; E[0] is the energy entering this medium
    E_beam: Optional[float] = None
    windows: list = field(default_factory=list)
    # energy straggling (core/straggling.py): var(E) = dEdx_mass^2 * strag_J at every point
    strag_J: Optional[np.ndarray] = None

    @property
    def sigma_E(self) -> Optional[np.ndarray]:
        """1-sigma energy spread (MeV) along the track, None without straggling"""
        if self.strag_J is None:
            return None
        return np.sqrt(np.maximum(self.dEdx_mass ** 2 * self.strag_J, 0.0))

    @property
    def range_sigma_mm(self) -> Optional[float]:
        """
        1-sigma range straggling (mm) from energy straggling; None without straggling or when
        the particle does not stop in this medium. A particle delta above the mean energy is
        delta / S further ahead, and J is the variance of exactly that distance (in g/cm^2)^2.
        """
        if self.strag_J is None or self.stop_reason != "stopped":
            return None
        return float(np.sqrt(self.strag_J[-1]) / self.rho * 10.0)

    @property
    def window_loss(self) -> float:
        """Energy lost in the windows (MeV), 0 without windows"""
        return 0.0 if self.E_beam is None else float(self.E_beam - self.E[0])

    @property
    def is_complete(self) -> bool:
        """False when the depth or step limit cut the track short"""
        return self.stop_reason in ("stopped", "model_limit", "window")

    @property
    def unspent_fraction(self) -> float:
        """Share of the entry energy still left at the end of the track (0 = fully slowed down)"""
        return float(self.E[-1] / self.E[0]) if self.E[0] > 0 else 0.0

    @property
    def x_mm(self) -> np.ndarray:
        return self.x_mass / self.rho * 10.0

    def dEdx_in(self, unit: StoppingUnit) -> np.ndarray:
        return convert_from_mass(self.dEdx_mass, unit, self.rho)

    def x_in(self, mass_thickness: bool = False) -> np.ndarray:
        return self.x_mass if mass_thickness else self.x_mm

    @property
    def range_mm(self) -> float:
        return float(self.x_mm[-1])

    @property
    def range_mass(self) -> float:
        return float(self.x_mass[-1])

    def peak_dEdx(self, unit: StoppingUnit) -> float:
        return float(np.nanmax(self.dEdx_in(unit)))


# ---- integrator ----

def _locate_model_limit(mat: Material, part: Particle, hi: float, lo: float) -> float:
    """
    Lowest energy in [lo, hi] where electronic stopping is still valid, given that
    it is valid at hi and not at lo
    """
    for _ in range(3):
        E = np.linspace(hi, lo, 101)
        bad = np.flatnonzero(dEdx_components(mat, part, E)[0] <= 0)
        if bad.size == 0:                 # valid all the way down to lo
            return float(lo)
        j = int(bad[0])
        hi, lo = E[j - 1], E[j]
    return float(hi)


def simulate(mat: Material, part: Particle, settings: SimSettings) -> TrackResult:
    """
    Slow the particle down on a logarithmic energy grid, from E0 to E_cutoff, and then
    to exactly E = 0.

    The stopping power is evaluated for the whole grid in one vectorized call and
    the depth follows from  x(E) = x_start + integral_E^E0 dE' / S(E')  (trapezoid
    rule in ln E). Below E_cutoff the stopping power is taken to fall like sqrt(E) (true
    for both the Lindhard-Scharff and the ZBL nuclear part at these energies), which gives
    the last bit of range as 2 E / S. The track is cut at the depth limit if one is set
    ("x_stop") and where electronic stopping is invalid ("model_limit").
    """
    if part.E0 is None or part.E0 <= 0:
        raise ValueError(f"{part.name}: initial kinetic energy must be > 0 MeV.")

    mm_to_mass = mat.rho / 10.0
    x_start_m  = settings.x_start * mm_to_mass
    x_stop_m   = None if settings.x_stop is None else settings.x_stop * mm_to_mass
    E0 = float(part.E0)
    E_floor = settings.E_cutoff

    def result(x, E, S, reason):
        return TrackResult(name=part.name, color=part.color,
                           x_mass=np.asarray(x, float), E=np.asarray(E, float),
                           dEdx_mass=np.asarray(S, float), rho=mat.rho,
                           stop_reason=reason)

    nothing = float("nan")
    if E0 <= E_floor:
        return result([x_start_m], [E0], [nothing], "stopped")

    n = max(2, int(np.ceil(np.log(E0 / E_floor) / settings.dlnE)) + 1)
    E = np.geomspace(E0, E_floor, n)                    # descending
    elec, nuc = dEdx_components(mat, part, E)
    S = elec + nuc

    stop_reason = "stopped"
    invalid = np.flatnonzero(elec <= 0)
    if invalid.size:
        cut = int(invalid[0])
        if cut == 0:
            return result([x_start_m], [E0], [nothing], "model_limit")
        # S is small at the limit
        # end the track at the limit itself, not at the last grid point above it
        E_lim = _locate_model_limit(mat, part, hi=E[cut - 1], lo=E[cut])
        S_lim = sum(dEdx_components(mat, part, E_lim))
        E = np.append(E[:cut], E_lim)
        S = np.append(S[:cut], S_lim)
        stop_reason = "model_limit"

    # depth: trapezoid rule for  integral (E / S) d(ln E)
    f = E / S
    lnE = np.log(E)
    seg = 0.5 * (f[1:] + f[:-1]) * (lnE[:-1] - lnE[1:])
    x = x_start_m + np.concatenate(([0.0], np.cumsum(seg)))

    # the particle reaches the energy cutoff
    if stop_reason == "stopped":
        x = np.append(x, x[-1] + 2.0 * E[-1] / S[-1])
        E = np.append(E, 0.0)
        S = np.append(S, 0.0)

    # depth limit: end the track exactly at x_stop
    if x_stop_m is not None and (x[0] >= x_stop_m or x[-1] > x_stop_m):
        k = int(np.searchsorted(x, x_stop_m))
        if k == 0:
            x, E, S = x[:1], E[:1], S[:1]
        else:
            E_c = np.interp(x_stop_m, x, E)
            S_c = np.interp(x_stop_m, x, S)
            x = np.append(x[:k], x_stop_m)
            E = np.append(E[:k], E_c)
            S = np.append(S[:k], S_c)
        stop_reason = "x_stop"

    return result(x, E, S, stop_reason)


# ---- intersection finder ----

def curve_crossings(xa: np.ndarray, ya: np.ndarray,
                    xb: np.ndarray, yb: np.ndarray) -> List[tuple]:
    """
    Points where two piecewise-linear curves cross, as (x, y) tuples
    """
    xa, ya = np.asarray(xa, float), np.asarray(ya, float)
    xb, yb = np.asarray(xb, float), np.asarray(yb, float)
    ka, kb = np.isfinite(ya), np.isfinite(yb) 
    xa, ya, xb, yb = xa[ka], ya[ka], xb[kb], yb[kb]
    if len(xa) < 2 or len(xb) < 2:
        return []

    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    if lo >= hi:
        return []

    grid = np.union1d(xa, xb)
    grid = grid[(grid >= lo) & (grid <= hi)]
    fa = np.interp(grid, xa, ya)
    fb = np.interp(grid, xb, yb)
    diff = fa - fb

    nonneg = diff >= 0
    crossings: List[tuple] = []
    for i in np.where(nonneg[:-1] != nonneg[1:])[0]:
        d0, d1 = diff[i], diff[i + 1]
        t = d0 / (d0 - d1)
        x = grid[i] + t * (grid[i + 1] - grid[i])
        y = fa[i] + t * (fa[i + 1] - fa[i])
        if not crossings or not np.isclose(x, crossings[-1][0], rtol=1e-9, atol=0.0):
            crossings.append((float(x), float(y)))
    return crossings


def find_intersections(a: TrackResult, b: TrackResult,
                       unit: StoppingUnit,
                       mass_thickness: bool = False) -> List[tuple]:
    """Crossings of the dE/dx curves of two tracks"""
    return curve_crossings(a.x_in(mass_thickness), a.dEdx_in(unit),
                           b.x_in(mass_thickness), b.dEdx_in(unit))
