"""
Small element table and chemical-formula parser

Used to split a compound (e.g. isobutane, C4H10) into its atoms, because the
low-energy stopping models (ZBL nuclear, Lindhard-Scharff electronic) work
per target ATOM, not per molecule.

Extend ELEMENTS if you add a material containing an element not listed here.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# symbol -> (Z, standard atomic weight in g/mol)
ELEMENTS: Dict[str, tuple[int, float]] = {
    "H": (1, 1.008),    "He": (2, 4.0026),  "Li": (3, 6.94),    "Be": (4, 9.0122),
    "B": (5, 10.81),    "C": (6, 12.011),   "N": (7, 14.007),   "O": (8, 15.999),
    "F": (9, 18.998),   "Ne": (10, 20.180), "Na": (11, 22.990), "Mg": (12, 24.305),
    "Al": (13, 26.982), "Si": (14, 28.086), "P": (15, 30.974),  "S": (16, 32.06),
    "Cl": (17, 35.45),  "Ar": (18, 39.948), "K": (19, 39.098),  "Ca": (20, 40.078),
    "Ti": (22, 47.867), "Fe": (26, 55.845), "Ni": (28, 58.693), "Cu": (29, 63.546),
    "Ge": (32, 72.630), "Kr": (36, 83.798), "Mo": (42, 95.95),  "Ag": (47, 107.87),
    "Sn": (50, 118.71), "Xe": (54, 131.29), "W": (74, 183.84),  "Au": (79, 196.97),
    "Pb": (82, 207.2), 
}

_FORMULA_RE = re.compile(r"(?:[A-Z][a-z]?\d*)+")
_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(text: str) -> Optional[Dict[str, int]]:
    """'C4H10' -> {'C': 4, 'H': 10}.  None if not a simple formula / unknown element"""
    text = (text or "").strip()
    if not text or not _FORMULA_RE.fullmatch(text):
        return None

    counts: Dict[str, int] = {}
    for sym, num in _TOKEN_RE.findall(text):
        if sym not in ELEMENTS:
            return None
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts
