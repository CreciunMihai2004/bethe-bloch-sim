"""
Reader for SRIM "Stopping / Range" output tables (SRIM-2008 / SRIM-2013)

SRIM is used here only as a benchmark: the app computes its own stopping powers.
The reader accepts both decimal points and decimal commas (SRIM writes the
separator of the Windows locale, e.g. "6,785E-01" on a Romanian system).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_E_UNITS = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}                 # -> MeV
_L_UNITS = {"A": 1e-7, "um": 1e-3, "mm": 1.0, "m": 1e3, "km": 1e6}           # -> mm
_S_UNITS = {                                                                 # -> MeV cm^2 / g
    "MeV / (mg/cm2)": 1e3,
    "keV / (ug/cm2)": 1e3,
    "keV / (mg/cm2)": 1.0,
    "eV / (1E15 atoms/cm2)": None,          # needs the target composition: not supported
}

_NUM = r"[-+]?\d+(?:[.,]\d*)?(?:[Ee][-+]?\d+)?"
_ROW = re.compile(
    rf"^\s*({_NUM})\s+(eV|keV|MeV|GeV)\s+({_NUM})\s+({_NUM})"
    rf"\s+({_NUM})\s+(A|um|mm|m|km)\s+({_NUM})\s+(A|um|mm|m|km)\s+({_NUM})\s+(A|um|mm|m|km)\s*$"
)


def _num(text: str) -> float:
    return float(text.replace(",", "."))


@dataclass
class SrimTable:
    ion: str
    ion_Z: int
    ion_mass_u: float
    density: float              # g/cm^3
    E: np.ndarray               # MeV
    S_elec: np.ndarray          # MeV cm^2 / g
    S_nuc: np.ndarray           # MeV cm^2 / g
    proj_range_mm: np.ndarray   # projected range (not path length)
    long_strag_mm: np.ndarray   # longitudinal (range) straggling, 1 sigma
    lat_strag_mm: np.ndarray    # lateral straggling, 1 sigma

    @property
    def S_total(self) -> np.ndarray:
        return self.S_elec + self.S_nuc


def read_srim_table(path: str | Path) -> SrimTable:
    text = Path(path).read_text(encoding="latin-1")

    m_ion = re.search(r"Ion\s*=\s*(\w+)\s*\[(\d+)\]\s*,\s*Mass\s*=\s*(" + _NUM + ")", text)
    m_rho = re.search(r"Target Density\s*=\s*(" + _NUM + r")\s*g/cm3", text)
    m_unit = re.search(r"Stopping Units\s*=\s*(.+?)\s*$", text, re.M)
    if not (m_ion and m_rho and m_unit):
        raise ValueError(f"{path}: not a SRIM stopping/range table (header not found)")

    unit = re.sub(r"\s+", " ", m_unit.group(1)).strip()
    factor = _S_UNITS.get(unit)
    if factor is None:
        raise ValueError(f"{path}: stopping unit '{unit}' is not supported; "
                         f"export the table in MeV/(mg/cm2) or keV/(ug/cm2)")

    rows = []
    for line in text.splitlines():
        r = _ROW.match(line)
        if r:
            g = r.groups()
            rows.append((_num(g[0]) * _E_UNITS[g[1]],
                         _num(g[2]) * factor, _num(g[3]) * factor,
                         _num(g[4]) * _L_UNITS[g[5]],
                         _num(g[6]) * _L_UNITS[g[7]],
                         _num(g[8]) * _L_UNITS[g[9]]))
    if not rows:
        raise ValueError(f"{path}: no data rows found")

    a = np.array(rows)
    return SrimTable(ion=m_ion.group(1), ion_Z=int(m_ion.group(2)),
                     ion_mass_u=_num(m_ion.group(3)), density=_num(m_rho.group(1)),
                     E=a[:, 0], S_elec=a[:, 1], S_nuc=a[:, 2], proj_range_mm=a[:, 3],
                     long_strag_mm=a[:, 4], lat_strag_mm=a[:, 5])
