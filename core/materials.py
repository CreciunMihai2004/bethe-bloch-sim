"""
Material definitions and database
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .data_io import get_data_dir, load_csv_db
from .elements import mean_excitation_energy, parse_formula

# a material whose density is below this is a gas: its I is built from gas-phase atomic values
GAS_MAX_DENSITY = 0.01      # g/cm^3

_DATA_DIR = get_data_dir()
_MATERIALS_CSV = _DATA_DIR / "materials.csv"
_WINDOWS_CSV = _DATA_DIR / "windows.csv"

@dataclass
class Material:
    """
    A target medium. With a chemical formula, Z, A and I may be left out (None): Z and A are
    summed over the formula and I follows from Bragg additivity (core.elements). A value that
    IS given is used as it stands (e.g. a measured I), and Z / A are checked against the formula.
    """
    name: str
    Z: float | None = None      # electrons per formula unit
    A: float | None = None      # g/mol per formula unit
    I_eV: float | None = None   # mean excitation energy (eV)
    rho: float | None = None    # g/cm^3

    # Sternheimer density-effect parameters
    X0: float = 1.6
    X1: float = 4.0
    a_stern: float = 0.10
    m_stern: float = 3.0

    # Chemical formula, e.g. "C4H10". Optional: the low-energy electronic stopping and the
    # nuclear stopping are summed over the elements (Bragg additivity).
    formula: str = ""

    # True when I_eV was computed from the formula rather than given
    I_from_formula: bool = False

    def __post_init__(self):
        if self.rho is None or not self.rho > 0:
            raise ValueError(f"'{self.name}': density must be > 0")
        if self.formula.strip():
            comp = parse_formula(self.formula)
            z_sum = sum(z * n for z, _, n in comp)
            a_sum = sum(a * n for _, a, n in comp)
            if self.Z is None:
                self.Z = z_sum
            elif abs(z_sum - self.Z) > 0.5:
                raise ValueError(
                    f"'{self.name}': formula {self.formula} has {z_sum:g} electrons per "
                    f"formula unit but the Z column says {self.Z:g}")
            if self.A is None:
                self.A = a_sum
            elif abs(a_sum - self.A) > 0.01 * a_sum:
                raise ValueError(
                    f"'{self.name}': formula {self.formula} weighs {a_sum:.3f} g/mol "
                    f"but the A column says {self.A:g}")
            if self.I_eV is None:
                self.I_eV = mean_excitation_energy(comp, gas=self.rho < GAS_MAX_DENSITY)
                self.I_from_formula = True
        missing = [k for k in ("Z", "A", "I_eV") if getattr(self, k) is None]
        if missing:
            raise ValueError(f"'{self.name}': give {', '.join(missing)} or a chemical formula")

    @property
    def composition(self) -> tuple[tuple[float, float, float], ...]:
        if self.formula.strip():
            return parse_formula(self.formula)
        return ((self.Z, self.A, 1.0),)

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

def _float(val: str | None, default: float | None) -> float | None:
    """Convert a CSV cell to float, using `default` when the cell is empty or missing"""
    s = val.strip() if val else ""
    return float(s) if s else default


def _build_material(row: dict) -> Material:
    return Material(
        name    = row["name"].strip(),
        Z       = _float(row.get("Z"),    None),     # blank: from the formula
        A       = _float(row.get("A"),    None),
        I_eV    = _float(row.get("I_eV"), None),
        rho     = float(row["rho"]),
        X0      = _float(row.get("X0"),      1.6),
        X1      = _float(row.get("X1"),      4.0),
        a_stern = _float(row.get("a_stern"), 0.10),
        m_stern = _float(row.get("m_stern"), 3.0),
        formula = (row.get("formula") or "").strip(),
    )

def _load_material_db(path: Path = _MATERIALS_CSV) -> dict[str, Material]:
    """
    Parse path and return a {name: Material} dict
    Lines whose first non-whitespace character is `#` are treated as
    comments and skipped
    """
    return load_csv_db(path, {"name", "rho"}, _build_material, "Material")


def _load_window_db(path: Path = _WINDOWS_CSV) -> dict[str, Material]:
    """Window / foil materials; the file is optional"""
    return _load_material_db(path) if path.exists() else {}


# detector media (materials.csv) and window materials (windows.csv)
MATERIAL_DB: dict[str, Material] = _load_material_db()
WINDOW_MATERIAL_DB: dict[str, Material] = _load_window_db()


def get_material(name: str) -> Material:
    for db in (MATERIAL_DB, WINDOW_MATERIAL_DB):
        if name in db:
            return replace(db[name])
    raise KeyError(
        f"Material '{name}' not found.\n"
        f"Available: {list(MATERIAL_DB) + list(WINDOW_MATERIAL_DB)}"
    )


def reload() -> None:
    for db, fresh in ((MATERIAL_DB, _load_material_db()), (WINDOW_MATERIAL_DB, _load_window_db())):
        db.clear()
        db.update(fresh)
