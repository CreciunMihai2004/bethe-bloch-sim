"""
Material definitions and database
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data_io import get_data_dir, load_csv_db
from .elements import ELEMENTS, parse_formula

_DATA_DIR = get_data_dir()
_MATERIALS_CSV = _DATA_DIR / "materials.csv"

# "Isobutane (C4H10)" -> "C4H10"
_NAME_FORMULA_RE = re.compile(r"\(([^()]+)\)\s*$")


@dataclass
class Material:
    name: str
    Z: float       # effective atomic number (for a compound: sum of Z over the molecule)
    A: float       # g/mol (for a compound: molar mass)
    I_eV: float    # mean excitation energy (eV)
    rho: float     # g/cm^3

    # Sternheimer density-effect parameters
    X0: float = 1.6
    X1: float = 4.0
    a_stern: float = 0.10
    m_stern: float = 3.0

    # Optional chemical formula ("C4H10").  When empty it is read from a
    # trailing "(...)" in the name.  Used only by the per-atom low-energy models
    formula: str = ""

    # Filled in __post_init__:
    #   components : ((Z_i, M_i, n_i), ...)  atoms making up one "molecule"
    #   mol_mass   : sum(n_i * M_i) in g/mol
    #   composition_source : "formula" or "single-atom"
    components: tuple = field(init=False, repr=False)
    mol_mass: float = field(init=False, repr=False)
    composition_source: str = field(init=False, repr=False)

    def __post_init__(self):
        self._resolve_composition()

    def _resolve_composition(self) -> None:
        formula = self.formula.strip()
        if not formula:
            m = _NAME_FORMULA_RE.search(self.name)
            formula = m.group(1) if m else ""

        counts = parse_formula(formula)
        if counts:
            comps = tuple((ELEMENTS[s][0], ELEMENTS[s][1], n) for s, n in counts.items())
            z_sum = sum(z * n for z, _, n in comps)
            m_sum = sum(m * n for _, m, n in comps)
            # accept the formula only if it agrees with the Z and A in the CSV
            if abs(z_sum - self.Z) <= 0.5 and abs(m_sum - self.A) <= 0.02 * self.A:
                self.components = comps
                self.mol_mass = m_sum
                self.composition_source = "formula"
                return

        # elementary material (or an unparseable name): one atom of (Z, A)
        self.components = ((self.Z, self.A, 1),)
        self.mol_mass = self.A
        self.composition_source = "single-atom"

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
        formula = (row.get("formula") or "").strip(),
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