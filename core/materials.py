"""
Material definitions and database
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_io import get_data_dir, load_csv_db

_DATA_DIR = get_data_dir()
_MATERIALS_CSV = _DATA_DIR / "materials.csv"

@dataclass
class Material:
    name: str
    Z: float
    A: float       # g/mol
    I_eV: float    # mean excitation energy (eV)
    rho: float     # g/cm^3

    # Sternheimer density-effect parameters
    X0: float = 1.6
    X1: float = 4.0
    a_stern: float = 0.10
    m_stern: float = 3.0

    @property
    def I_MeV(self) -> float:
        return self.I_eV * 1e-6

    @property
    def plasma_energy_eV(self) -> float:
        """Sternheimer plasma energy hbar*omega_p in eV"""
        return 28.816 * np.sqrt(self.rho * self.Z / self.A)

    @property
    def Cbar(self) -> float:
        """Sternheimer C-bar constant for the density-effect correction"""
        return 2.0 * np.log(self.I_eV / self.plasma_energy_eV) + 1.0

# ---- CSV loader ----

def _float(val: str | None, default: float) -> float:
    """Convert a CSV cell to float, using `default` when the cell is empty or missing"""
    s = val.strip() if val else ""
    return float(s) if s else default


def _build_material(row: dict) -> Material:
    return Material(
        name    = row["name"].strip(),
        Z       = float(row["Z"]),
        A       = float(row["A"]),
        I_eV    = float(row["I_eV"]),
        rho     = float(row["rho"]),
        X0      = _float(row.get("X0"),      1.6),
        X1      = _float(row.get("X1"),      4.0),
        a_stern = _float(row.get("a_stern"), 0.10),
        m_stern = _float(row.get("m_stern"), 3.0),
    )

def _load_material_db(path: Path = _MATERIALS_CSV) -> dict[str, Material]:
    """
    Parse path and return a {name: Material} dict
    Lines whose first non-whitespace character is `#` are treated as
    comments and skipped
    """
    return load_csv_db(path, {"name", "Z", "A", "I_eV", "rho"}, _build_material, "Material")

MATERIAL_DB: dict[str, Material] = _load_material_db()


def get_material(name: str) -> Material:
    try:
        return MATERIAL_DB[name]
    except KeyError:
        raise KeyError(
            f"Material '{name}' not found.\n"
            f"Available: {list(MATERIAL_DB)}"
        ) from None

def reload() -> None:
    """Re-read the CSV and refresh MATERIAL_DB in-place"""
    global MATERIAL_DB
    MATERIAL_DB = _load_material_db()