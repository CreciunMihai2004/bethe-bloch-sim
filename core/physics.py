import math

import numpy as np

from .constants import K, ME_C2, N_A
from .materials import Material
from .particles import Particle

def _out(a):
    """0-d array -> float"""
    a = np.asarray(a)
    return float(a) if a.ndim == 0 else a


def density_effect(beta, gamma, mat: Material):
    """Sternheimer density-effect correction delta(X), X = log10(beta*gamma)"""
    bg = np.asarray(beta, dtype=float) * np.asarray(gamma, dtype=float)
    X = np.log10(np.maximum(bg, 1e-300))
    delta_hi = 4.6052 * X - mat.Cbar
    delta_mid = delta_hi + mat.a_stern * np.clip(mat.X1 - X, 0.0, None) ** mat.m_stern
    delta = np.where(X < mat.X0, 0.0, np.where(X < mat.X1, delta_mid, delta_hi))
    return _out(np.where(bg <= 0, 0.0, delta))


def effective_charge(z: int, beta):
    """
    Effective charge
    """
    beta = np.asarray(beta, dtype=float)
    z_eff = z * (1.0 - np.exp(-125.0 * beta * z ** (-2.0 / 3.0)))
    return _out(np.where(beta > 0, z_eff, 0.0))

def zbl_reduced_energy(E_keV, Z1: float, Z2: float, M1: float, M2: float):
    """
    ZBL reduced energy epsilon (universal screening length)
    """
    return 32.53 * M2 * E_keV / (Z1 * Z2 * (M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))


def zbl_reduced_stopping(eps):
    """
    ZBL universal reduced nuclear stopping s_n(eps)
    """
    if np.ndim(eps) == 0:
        e = float(eps)
        if e <= 30.0:
            return math.log(1.0 + 1.1383 * e) / (
                2.0 * (e + 0.01321 * e ** 0.21226 + 0.19593 * e ** 0.5)
            )
        return math.log(e) / (2.0 * e)

    eps = np.asarray(eps, dtype=float)
    low = np.minimum(eps, 30.0)
    s_low = np.log(1.0 + 1.1383 * low) / (
        2.0 * (low + 0.01321 * low ** 0.21226 + 0.19593 * low ** 0.5)
    )
    s_high = np.log(np.maximum(eps, 1.0)) / (2.0 * np.maximum(eps, 1.0))
    return np.where(eps <= 30.0, s_low, s_high)


def zbl_nuclear_stopping(E_kin, part: Particle, mat: Material):
    """
    Nuclear mass stopping power (MeV cm^2 / g) from the ZBL universal potential
    (Ziegler, Biersack & Littmark 1985), summed over the elements of the target
    """
    E = np.asarray(E_kin, dtype=float)
    live = E > 1e-6
    Es = np.where(live, E, 1.0)         # safe value in dead slots, zeroed at the end

    Z1, M1 = part.z, part.M_u

    # absolute stopping cross section per formula unit, eV / (1e15 atoms / cm^2)
    S_n = 0.0
    for Z2, M2, n in mat.composition:
        eps = zbl_reduced_energy(Es * 1000.0, Z1, Z2, M1, M2)
        s_n = zbl_reduced_stopping(eps)
        S_n = S_n + n * (8.462 * Z1 * Z2 * M1 * s_n) / ((M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))

    # mass stopping power: x (N_A / A) formula units/g, x 1e-15 (per 1e15 atoms), x 1e-6 (eV -> MeV)
    mass = np.maximum(S_n * (N_A / mat.A) * 1e-21, 0.0)
    return _out(np.where(live, mass, 0.0))


def bethe_stopping(mat: Material, part: Particle, E_kin):
    """
    Bethe-Bloch electronic mass stopping power (MeV cm^2 / g) with effective charge and
    density-effect correction; 0 where the formula is not valid (its logarithm is below ~1)
    """
    E = np.asarray(E_kin, dtype=float)
    live = E > 1e-6
    Es = np.where(live, E, 1.0)

    M_part = part.M

    gamma = Es / M_part + 1.0
    beta_sq = np.maximum(1.0 - 1.0 / gamma ** 2, 1e-12)
    beta = np.sqrt(beta_sq)

    z_eff = np.asarray(effective_charge(part.z, beta))

    ratio = ME_C2 / M_part
    T_max = (2.0 * ME_C2 * beta_sq * gamma ** 2) / (
        1.0 + 2.0 * gamma * ratio + ratio ** 2
    )
    log_arg = (2.0 * ME_C2 * beta_sq * gamma ** 2 * T_max) / (mat.I_MeV ** 2)

    # Using 1.05 as a buffer so it switches before hitting 0
    ok = (z_eff > 0) & (log_arg > 1.05)
    delta = density_effect(beta, gamma, mat)

    bb_stopping = K * (z_eff ** 2) * (mat.Z / mat.A) * (1.0 / beta_sq) * (
        0.5 * np.log(np.where(ok, log_arg, math.e)) - beta_sq - delta / 2.0
    )
    return _out(np.where(ok & live, np.maximum(bb_stopping, 0.0), 0.0))


# Below E_m the electronic stopping follows the velocity-proportional law S ~ sqrt(E)
# (Lindhard-Scharff), above it Bethe-Bloch. E_m is where the Bethe-Bloch curve's own
# logarithmic slope has fallen to 1/2, the slope of that law, so the two meet with the same
# value and the same slope and no free parameter. The amplitude comes from Bethe-Bloch itself.
_MATCH_CACHE: dict[tuple, tuple[float, float]] = {}
_LS_EXPONENT = 0.5


def _matching_point(mat: Material, part: Particle) -> tuple[float, float]:
    """
    (E_m in MeV, Bethe-Bloch stopping at E_m): the join between the velocity-proportional
    law below and Bethe-Bloch above. (0.0, 0.0) means Bethe-Bloch only, for a combination
    where the curve never reaches that slope.
    """
    key = (mat.Z, mat.A, mat.I_eV, mat.rho, mat.X0, mat.X1, mat.a_stern, mat.m_stern,
           part.z, part.M_u)
    hit = _MATCH_CACHE.get(key)
    if hit is not None:
        return hit

    E = np.geomspace(1e-3, 1e4, 4000)
    S = np.asarray(bethe_stopping(mat, part, E))
    valid = S > 0
    slope = np.full(E.shape, np.inf)
    slope[1:] = np.diff(np.log(np.where(valid, S, np.nan))) / np.diff(np.log(E))
    flat = np.flatnonzero(valid & (slope <= _LS_EXPONENT))
    found = (float(E[flat[0]]), float(S[flat[0]])) if flat.size else (0.0, 0.0)

    if len(_MATCH_CACHE) > 256:
        _MATCH_CACHE.clear()
    _MATCH_CACHE[key] = found
    return found


def dEdx_components(mat: Material, part: Particle, E_kin):
    """
    Mass stopping power split into its two parts (MeV cm^2 / g):

        (electronic, nuclear)
    """
    E = np.asarray(E_kin, dtype=float)
    live = E > 1e-6
    Es = np.where(live, E, 1.0)         # safe value in dead slots, zeroed at the end

    bethe = np.asarray(bethe_stopping(mat, part, Es))
    E_m, S_m = _matching_point(mat, part)
    low = S_m * (Es / E_m) ** _LS_EXPONENT if E_m > 0 else 0.0
    electronic = np.where(Es < E_m, low, bethe)

    nuclear = np.asarray(zbl_nuclear_stopping(Es, part, mat))

    return _out(np.where(live, electronic, 0.0)), _out(np.where(live, nuclear, 0.0))


def dEdx_mass(mat: Material, part: Particle, E_kin):
    """
    TOTAL MASS stopping power (Electronic + Nuclear)
    """
    elec, nuc = dEdx_components(mat, part, E_kin)
    return elec + nuc
