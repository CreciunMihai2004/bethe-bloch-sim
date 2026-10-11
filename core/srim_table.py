"""
Reader for SRIM "Range" output tables (the .txt files SRIM writes when you
calculate a stopping/range table for one ion in one target).

A table is a reference for ONE ion in ONE target.  Mass stopping power
(MeV cm^2/g) does not depend on the density SRIM was run at, so a table
made at any gas density is valid at any pressure.

The parser handles:
  * decimal commas ("6,785E-01") as written by a Romanian-locale SRIM
  * energy units eV / keV / MeV / GeV and range units A / um / mm / m / km
  * any "Stopping Units" choice: the conversion row for MeV/(mg/cm2) at the
    bottom of the table is used, so the units you picked in SRIM do not matter
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NUM = r"[\d.,]+"
_SCI = r"[\d.,]+[Ee][+-]?\d+"

_ROW_RE = re.compile(
    rf"^\s*({_NUM})\s+(eV|keV|MeV|GeV)\s+({_SCI})\s+({_SCI})\s+({_NUM})\s+(A|um|mm|m|km)\b"
)
_FACTOR_RE = re.compile(rf"({_SCI})\s+MeV\s*/\s*\(mg/cm2\)")
_DENSITY_RE = re.compile(rf"Target Density\s*=\s*({_SCI})\s*g/cm3")
_ION_RE = re.compile(r"Ion\s*=\s*(\w+)\s*\[(\d+)\]\s*,\s*Mass\s*=\s*([\d.,]+)\s*amu")

_E_TO_MEV = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}
_R_TO_MM = {"A": 1e-7, "um": 1e-3, "mm": 1.0, "m": 1e3, "km": 1e6}


def _f(text: str) -> float:
    return float(text.replace(",", "."))


@dataclass
class SrimTable:
    """
    E       : ion energy (MeV), ascending
    Se, Sn  : electronic / nuclear MASS stopping power (MeV cm^2 / g)
    Rp_mm   : projected range at density `density` (mm)
    density : target density SRIM used (g/cm^3), for the range column only
    """
    E: np.ndarray
    Se: np.ndarray
    Sn: np.ndarray
    Rp_mm: np.ndarray
    density: float
    ion: str = ""
    Z1: int = 0
    mass_u: float = 0.0
    source: str = ""

    @property
    def E_min(self) -> float:
        return float(self.E[0])

    @property
    def E_max(self) -> float:
        return float(self.E[-1])

    def Se_at(self, E_MeV: float) -> float:
        """Electronic stopping inside the table range (log-log interpolation)"""
        return float(np.exp(np.interp(np.log(E_MeV), np.log(self.E), np.log(self.Se))))

    def Sn_at(self, E_MeV: float) -> float:
        return float(np.exp(np.interp(np.log(E_MeV), np.log(self.E), np.log(self.Sn))))

    def csda_mm(self, density: float | None = None) -> float:
        """
        Path length from E_min to E_max obtained by integrating 1/(Se+Sn) over
        the table (mm, at `density`).  Does NOT include the range below E_min
        """
        S = self.Se + self.Sn
        mass = float(np.sum(0.5 * (1.0 / S[1:] + 1.0 / S[:-1]) * np.diff(self.E)))
        return mass / (density or self.density) * 10.0


def parse_srim_table(path: str | Path) -> SrimTable:
    path = Path(path)
    text = path.read_text(encoding="latin-1")

    factor = _FACTOR_RE.search(text)
    if not factor:
        raise ValueError(
            f"{path.name}: no 'MeV / (mg/cm2)' conversion row found; "
            "is this a SRIM range table?"
        )
    to_mev_cm2_per_g = _f(factor.group(1)) * 1000.0      # value * factor = MeV/(mg/cm2)

    E, Se, Sn, Rp = [], [], [], []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m:
            E.append(_f(m[1]) * _E_TO_MEV[m[2]])
            Se.append(_f(m[3]) * to_mev_cm2_per_g)
            Sn.append(_f(m[4]) * to_mev_cm2_per_g)
            Rp.append(_f(m[5]) * _R_TO_MM[m[6]])
    if len(E) < 3:
        raise ValueError(f"{path.name}: found only {len(E)} table rows")

    order = np.argsort(E)
    dens = _DENSITY_RE.search(text)
    ion = _ION_RE.search(text)
    return SrimTable(
        E=np.array(E)[order], Se=np.array(Se)[order], Sn=np.array(Sn)[order],
        Rp_mm=np.array(Rp)[order],
        density=_f(dens.group(1)) if dens else 0.0,
        ion=ion.group(1) if ion else "",
        Z1=int(ion.group(2)) if ion else 0,
        mass_u=_f(ion.group(3)) if ion else 0.0,
        source=path.name,
    )
