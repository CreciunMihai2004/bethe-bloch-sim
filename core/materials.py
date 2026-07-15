"""
Material definitions
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Material:
    name: str
    Z: float
    A: float                # g/mol
    I_eV: float             # mean excitation energy (eV)
    rho: float              # g/cm^3

    # Sternheimer density-effect parameters
    X0: float = 1.6
    X1: float = 4.0
    a_stern: float = 0.10
    m_stern: float = 3.0

    @property
    def I_MeV(self) -> float:
        """Mean excitation energy in MeV"""
        return self.I_eV * 1e-6

    @property
    def plasma_energy_eV(self) -> float:
        """Sternheimer plasma energy hbar*omega_p in eV"""
        return 28.816 * np.sqrt(self.rho * self.Z / self.A)

    @property
    def Cbar(self) -> float:
        """Sternheimer C-bar constant for the density-effect correction"""
        return 2.0 * np.log(self.I_eV / self.plasma_energy_eV) + 1.0


# Preset database 
MATERIAL_DB = {
    "Isobutane (C4H10)": Material(
        name="Isobutane (C4H10)",
        Z=34.0,
        A=58.124,       # g/mol
        I_eV=48.3,      # eV
        rho=2.51,       # g/cm^3
    ),
    # To add more
}


def get_material(name: str) -> Material:
    return MATERIAL_DB[name]