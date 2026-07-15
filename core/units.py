"""
Stopping-power output units
"""

from enum import Enum


class StoppingUnit(Enum):
    MASS = "MeV cm^2 / g"      # mass stopping power
    LINEAR_MM = "MeV / mm"     # linear stopping power
    LINEAR_CM = "MeV / cm"     # linear stopping power

    @property
    def label(self) -> str:
        return self.value


def convert_from_mass(mass_stopping: float, unit: StoppingUnit, rho: float) -> float:
    
    if unit is StoppingUnit.MASS:
        return mass_stopping
    if unit is StoppingUnit.LINEAR_CM:
        return mass_stopping * rho
    if unit is StoppingUnit.LINEAR_MM:
        return mass_stopping * rho / 10.0
    raise ValueError(f"Unknown unit: {unit}")