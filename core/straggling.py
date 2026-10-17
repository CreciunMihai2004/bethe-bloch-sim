"""
Energy-loss straggling (Gaussian, first order)

Rate: Bohr's formula per unit mass thickness,
    dOmega^2/dX = K m_e c^2 z_eff^2 * sum_i (n_i Z_i / A) f(chi_i)          [MeV^2 cm^2 / g]
with the same effective charge as the stopping power, summed over the elements of the target
(Bragg additivity), and the Lindhard-Scharff low-velocity factor
    f(chi) = min(1, L(chi) / 2),  L = 1.36 chi^1/2 - 0.016 chi^3/2,  chi = (v / v0)^2 / Z_i
which switches the bound electrons off as the ion slows down (taken as 1 once L/2 reaches 1,
at chi ~ 2.3, so the factor is continuous and never exceeds Bohr).

Not included yet: charge-exchange straggling (fluctuations of the ion's charge state, which
widen the distribution for heavy ions in gases) and nuclear-collision straggling.

Propagation: a particle that is delta above the mean energy keeps delta / S constant (it is
a fixed distance ahead in range), and every layer dX adds its own variance. To first order

    var E(X) = S(X)^2 * J(X),   J(X) = var0 / S(X0)^2 + integral_X0^X (dOmega^2/dX') / S(X')^2 dX'

with S the mass stopping power. u = dE / S is a random walk with independent steps, so
    cov(E(X1), E(X2)) = S(X1) S(X2) J(min(X1, X2))
and J at the end of a track that stops is the range-straggling variance in (g/cm^2)^2.
J is stored on the track (TrackResult.strag_J).
"""

from __future__ import annotations

import numpy as np

from .constants import ALPHA_INV, K, ME_C2
from .materials import Material
from .particles import Particle
from .physics import dEdx_mass, effective_charge

_BOHR = K * ME_C2      # 4 pi e^4 N_A in MeV^2 cm^2 / mol


def _ls_switch() -> float:
    """chi where L(chi)/2 first reaches 1 (L is a fit for small chi only: it turns over later)"""
    # with s = sqrt(chi):  -0.016 s^3 + 1.36 s - 2 = 0, smallest positive root
    s = np.roots([-0.016, 0.0, 1.36, -2.0])
    s = min(r.real for r in s if abs(r.imag) < 1e-12 and r.real > 0)
    return float(s * s)


_CHI_SWITCH = _ls_switch()      # ~2.2


def lindhard_scharff_factor(chi):
    """Fraction of a target element's electrons that take part (Bohr = 1)"""
    chi = np.asarray(chi, dtype=float)
    L = 1.36 * np.sqrt(chi) - 0.016 * chi ** 1.5
    return np.where(chi >= _CHI_SWITCH, 1.0, np.clip(0.5 * L, 0.0, 1.0))


def straggling_rate(mat: Material, part: Particle, E_kin) -> np.ndarray:
    """dOmega^2/dX in MeV^2 per g/cm^2 at kinetic energy E_kin (MeV), array in, array out"""
    E = np.atleast_1d(np.asarray(E_kin, dtype=float))
    Es = np.maximum(E, 1e-12)
    gamma = Es / part.M + 1.0
    beta_sq = np.maximum(1.0 - 1.0 / gamma ** 2, 0.0)
    z_eff = np.asarray(effective_charge(part.z, np.sqrt(beta_sq)))
    v_over_v0_sq = beta_sq * ALPHA_INV ** 2

    electrons = 0.0
    for Z2, _A2, n in mat.composition:
        electrons = electrons + n * Z2 * lindhard_scharff_factor(v_over_v0_sq / Z2)
    rate = _BOHR * z_eff ** 2 * electrons / mat.A
    return np.where(E > 0, rate, 0.0)


def reduced_variance(x_mass: np.ndarray, S_mass: np.ndarray, rate: np.ndarray,
                     var0: float) -> np.ndarray:
    """
    J along a track (see module docstring). x_mass ascending, S_mass > 0 except possibly at
    the final point where the particle stops (J is carried over there).
    """
    S = np.asarray(S_mass, float)
    ok = S > 0
    g = np.where(ok, rate / np.where(ok, S, 1.0) ** 2, 0.0)
    if not ok[-1] and len(g) > 1:
        g[-1] = g[-2]                   # rate / S^2 ~ sqrt(E) -> finite: take the last value
    J0 = var0 / S[0] ** 2 if S[0] > 0 else 0.0
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(x_mass)
    return J0 + np.concatenate(([0.0], np.cumsum(seg)))


def track_reduced_variance(track, mat: Material, part: Particle, var0: float) -> np.ndarray:
    return reduced_variance(track.x_mass, track.dEdx_mass, straggling_rate(mat, part, track.E), var0)


def energy_covariance(track, depths_mm) -> np.ndarray:
    """
    Covariance matrix (MeV^2) of the energies at the given depths (mm from the start of the
    track). Beyond the end of the range the energy is 0 with no spread.
    """
    if track.strag_J is None:
        raise ValueError("the track was computed without straggling")
    x = track.x_mm - track.x_mm[0]
    d = np.asarray(depths_mm, dtype=float)
    S = np.interp(d, x, track.dEdx_mass, right=0.0)
    if track.stop_reason != "stopped":
        S = np.where(d > x[-1], 0.0, S)          # no information past a cut track
    J = np.interp(d, x, track.strag_J)
    Jmin = np.minimum.outer(J, J)                 # J is monotonic in depth
    return np.outer(S, S) * Jmin


def range_straggling_mass(mat: Material, part: Particle, E0: float, var0: float = 0.0,
                          n: int = 4000) -> float:
    """
    1-sigma range straggling in g/cm^2, by direct integration (independent of the track
    grid; used to check J):  sigma_R^2 = var0 / S(E0)^2 + integral_0^E0 (dOmega^2/dX) / S^3 dE
    """
    E = np.geomspace(max(E0 * 1e-6, 2e-6), E0, n)          # the model is 0 below 1 eV
    S = np.asarray(dEdx_mass(mat, part, E))
    f = straggling_rate(mat, part, E) / S ** 3
    # integrate in ln E: dE = E dlnE
    return float(np.sqrt(var0 / S[-1] ** 2 + np.trapezoid(f * E, np.log(E))))
