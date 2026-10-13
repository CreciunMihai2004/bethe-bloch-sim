import math

import numpy as np

from .constants import K, ME_C2, N_A
from .materials import Material
from .particles import Particle


def density_effect(beta: float, gamma: float, mat: Material) -> float:
    """Sternheimer density-effect correction delta(X), X = log10(beta*gamma)"""
    bg = beta * gamma
    if bg <= 0:
        return 0.0
    X = np.log10(bg)
    if X < mat.X0:
        return 0.0
    elif X < mat.X1:
        return 4.6052 * X - mat.Cbar + mat.a_stern * (mat.X1 - X) ** mat.m_stern
    else:
        return 4.6052 * X - mat.Cbar


def effective_charge(z: int, beta: float) -> float:
    """
    Effective charge
    """
    if beta <= 0:
        return 0.0
    return z * (1.0 - np.exp(-125.0 * beta * z ** (-2.0 / 3.0)))

def zbl_reduced_energy(E_keV: float, Z1: float, Z2: float, M1: float, M2: float) -> float:
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


def zbl_nuclear_stopping(E_kin: float, part: Particle, mat: Material) -> float:
    """
    Nuclear mass stopping power (MeV cm^2 / g) from the ZBL universal potential
    (Ziegler, Biersack & Littmark 1985).
    """
    if E_kin <= 1e-6:
        return 0.0

    Z1, M1 = part.z, part.M_u
    Z2, M2 = mat.Z, mat.A

    eps = zbl_reduced_energy(E_kin * 1000.0, Z1, Z2, M1, M2)
    s_n = zbl_reduced_stopping(eps)

    # absolute stopping cross section, eV / (1e15 atoms / cm^2)
    S_n = (8.462 * Z1 * Z2 * M1 * s_n) / ((M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))

    # mass stopping power: x (N_A / M2) atoms/g, x 1e-15 (per 1e15 atoms), x 1e-6 (eV -> MeV)
    return max(S_n * (N_A / M2) * 1e-21, 0.0)


def dEdx_components(mat: Material, part: Particle, E_kin: float) -> tuple[float, float]:
    """
    Mass stopping power split into its two parts (MeV cm^2 / g):

        (electronic, nuclear)
    """
    if E_kin <= 1e-6:
        return 0.0, 0.0

    z_part = part.z
    M_part = part.M

    gamma = (E_kin / M_part) + 1.0
    beta_sq = 1.0 - (1.0 / gamma ** 2)
    beta_sq = max(beta_sq, 1e-12)
    beta = np.sqrt(beta_sq)

    # Nuclear Stopping (ZBL)
    mass_stopping_nuc = zbl_nuclear_stopping(E_kin, part, mat)

    # Electronic Stopping
    mass_stopping_elec = 0.0
    z_eff = effective_charge(z_part, beta)

    if z_eff > 0:
        ratio = ME_C2 / M_part
        T_max = (2.0 * ME_C2 * beta_sq * gamma ** 2) / (
            1.0 + 2.0 * gamma * ratio + ratio ** 2
        )

        I_MeV = mat.I_MeV
        log_arg = (2.0 * ME_C2 * beta_sq * gamma ** 2 * T_max) / (I_MeV ** 2)

        # Using 1.05 as a buffer so it switches before hitting 0
        if log_arg > 1.05:
            delta = density_effect(beta, gamma, mat)

            bb_stopping = K * (z_eff ** 2) * (mat.Z / mat.A) * (1.0 / beta_sq) * (
                0.5 * np.log(log_arg) - beta_sq - delta / 2.0
            )
            mass_stopping_elec = max(bb_stopping, 0.0)

    return mass_stopping_elec, mass_stopping_nuc


def dEdx_mass(mat: Material, part: Particle, E_kin: float) -> float:
    """
    TOTAL MASS stopping power (Electronic + Nuclear)
    """
    elec, nuc = dEdx_components(mat, part, E_kin)
    return elec + nuc
