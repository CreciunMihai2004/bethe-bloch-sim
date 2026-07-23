"""
Bethe-Bloch stopping-power
"""

import numpy as np

from .constants import K, ME_C2
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


def shell_correction(eta: float, mat: Material) -> float:
    """
    Barkas shell correction
    eta = beta*gamma. Valid only for eta >~ 0.13
    """
    if eta < 0.13:
        return 0.0
    I_eV = mat.I_eV
    eta2 = eta ** 2
    eta4 = eta2 ** 2
    eta6 = eta2 ** 3
    term1 = (0.422377 / eta2 + 0.0304043 / eta4 - 0.00038106 / eta6) * 1e-6 * I_eV ** 2
    term2 = (3.850190 / eta2 - 0.1667989 / eta4 + 0.00157955 / eta6) * 1e-9 * I_eV ** 3
    C = term1 + term2
    return max(C, 0.0)


def effective_charge(z: int, beta: float) -> float:
    """
    Ziegler/Brandt-Kitagawa-type effective charge
    """
    if beta <= 0:
        return 0.0
    return z * (1.0 - np.exp(-125.0 * beta * z ** (-2.0 / 3.0)))

def zbl_nuclear_stopping(E_kin: float, part: Particle, mat: Material) -> float:
    """
    Calculates the nuclear mass stopping power using the ZBL (Ziegler-Biersack-Littmark) universal formula
    """
    if E_kin <= 1e-6:
        return 0.0

    E_keV = E_kin * 1000.0
    
    Z1 = part.z
    M1 = part.M_u
    Z2 = mat.Z
    M2 = mat.A
    
    # Calculate reduced energy (epsilon)
    denominator = Z1 * Z2 * (M1 + M2) * (Z1**0.23 + Z2**0.23)
    epsilon = (32.53 * M2 * E_keV) / denominator
    
    # Calculate reduced nuclear stopping cross-section (s_n)
    if epsilon < 30.0:
        num = 0.5 * np.log(1.0 + 1.1383 * epsilon)
        den = epsilon + 0.01321 * (epsilon**0.21226) + 0.19593 * (epsilon**0.5)
        s_n = num / den
    else:
        s_n = np.log(epsilon) / (2.0 * epsilon)
        
    # Convert to physical cross section S_n
    S_n_eV_cm2 = (8.462 * Z1 * Z2 * M1 * s_n) / ((M1 + M2) * (Z1**0.23 + Z2**0.23))
    
    # Convert to Mass Stopping Power (MeV cm^2 / g)
    N_A = 6.022e23 # Avogadro's number
    mass_stopping_power = S_n_eV_cm2 * (N_A / M2) * 1e-21
    
    return max(mass_stopping_power, 0.0)

def dEdx_mass(mat: Material, part: Particle, E_kin: float) -> float:
    """
    TOTAL MASS stopping power in MeV cm^2 / g (Electronic + Nuclear)
    """
    if E_kin <= 1e-6:
        return 0.0
        
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
    
    # Flags to check if Bethe-Bloch is valid
    bb_valid = False
    
    if z_eff > 0:
        ratio = ME_C2 / M_part
        T_max = (2.0 * ME_C2 * beta_sq * gamma ** 2) / (
            1.0 + 2.0 * gamma * ratio + ratio ** 2
        )

        I_MeV = mat.I_MeV
        log_arg = (2.0 * ME_C2 * beta_sq * gamma ** 2 * T_max) / (I_MeV ** 2)
        
        # Using 1.05 as a buffer so it switches before hitting 0
        if log_arg > 1.05: 
            bb_valid = True
            eta = beta * gamma
            C_over_Z = shell_correction(eta, mat) / mat.Z
            delta = density_effect(beta, gamma, mat)

            bb_stopping = K * (z_eff ** 2) * (mat.Z / mat.A) * (1.0 / beta_sq) * (
                0.5 * np.log(log_arg) - beta_sq - delta / 2.0 - C_over_Z
            )
            mass_stopping_elec = max(bb_stopping, 0.0)

    # Low Energy Fallback
    # If Bethe-Bloch breaks down (log_arg <= 1.05), low-energy regime is forced
    if not bb_valid:
        k_drag = 0.5 * (part.z ** 0.5) * (mat.Z / mat.A)
        mass_stopping_elec = k_drag * np.sqrt(E_kin * 1000.0) 

    # Total Stopping Power
    return mass_stopping_elec + mass_stopping_nuc