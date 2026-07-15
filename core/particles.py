"""
Projectile definitions and database
"""

from dataclasses import dataclass
from typing import Optional

from .constants import U_TO_MEV


@dataclass
class Particle:
    name: str
    z: int          # charge number
    M_u: float      # rest mass in atomic mass units (aum)
    E0: Optional[float] = None  # initial kinetic energy (MeV)
    color: str = "cyan"   # plot color hint

    @property
    def M(self) -> float:
        """Rest mass energy in MeV"""
        return self.M_u * U_TO_MEV

# Presets
PARTICLE_DB = {
    "Proton": Particle(name="Proton", z=1, M_u=1.00728, color="orange"),
    "Alpha":  Particle(name="Alpha",  z=2, M_u=4.00151, color="cyan"),
    
    "Beryllium": Particle("Beryllium", z=4, M_u=9.012, color="cyan"),
    "Boron": Particle("Boron", z=5, M_u=10.81, color="orange"),
}


def get_particle(name: str) -> Particle:
    return PARTICLE_DB[name]
