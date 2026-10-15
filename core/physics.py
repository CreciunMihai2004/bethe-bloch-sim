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
    ZBL universal reduced nuclear stopping s_n(eps)  (works on scalars and arrays)
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
    (Ziegler, Biersack & Littmark 1985).
    """
    E = np.asarray(E_kin, dtype=float)
    live = E > 1e-6
    Es = np.where(live, E, 1.0)         # safe value in dead slots, zeroed at the end

    Z1, M1 = part.z, part.M_u
    Z2, M2 = mat.Z, mat.A

    eps = zbl_reduced_energy(Es * 1000.0, Z1, Z2, M1, M2)
    s_n = zbl_reduced_stopping(eps)

    # absolute stopping cross section, eV / (1e15 atoms / cm^2)
    S_n = (8.462 * Z1 * Z2 * M1 * s_n) / ((M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))

    # mass stopping power: x (N_A / M2) atoms/g, x 1e-15 (per 1e15 atoms), x 1e-6 (eV -> MeV)
    mass = np.maximum(S_n * (N_A / M2) * 1e-21, 0.0)
    return _out(np.where(live, mass, 0.0))


def dEdx_components(mat: Material, part: Particle, E_kin):
    """
    Mass stopping power split into its two parts (MeV cm^2 / g):

        (electronic, nuclear)
    """
    E = np.asarray(E_kin, dtype=float)
    live = E > 1e-6
    Es = np.where(live, E, 1.0)         # safe value in dead slots, zeroed at the end

    M_part = part.M

    gamma = Es / M_part + 1.0
    beta_sq = np.maximum(1.0 - 1.0 / gamma ** 2, 1e-12)
    beta = np.sqrt(beta_sq)

    # Nuclear Stopping (ZBL)
    mass_stopping_nuc = zbl_nuclear_stopping(Es, part, mat)

    # Electronic Stopping
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
    mass_stopping_elec = np.where(ok & live, np.maximum(bb_stopping, 0.0), 0.0)

    return _out(mass_stopping_elec), _out(np.where(live, mass_stopping_nuc, 0.0))


def dEdx_mass(mat: Material, part: Particle, E_kin):
    """
    TOTAL MASS stopping power (Electronic + Nuclear)
    """
    elec, nuc = dEdx_components(mat, part, E_kin)
    return elec + nuc
