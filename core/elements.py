"""
Chemical elements and formula parsing (used for Bragg additivity)
"""

from __future__ import annotations

import re
from functools import lru_cache

# symbol -> (Z, standard atomic weight in g/mol)
ELEMENTS: dict[str, tuple[int, float]] = {
    "H": (1, 1.008),    "He": (2, 4.0026),  "Li": (3, 6.94),    "Be": (4, 9.0122),
    "B": (5, 10.81),    "C": (6, 12.011),   "N": (7, 14.007),   "O": (8, 15.999),
    "F": (9, 18.998),   "Ne": (10, 20.180), "Na": (11, 22.990), "Mg": (12, 24.305),
    "Al": (13, 26.982), "Si": (14, 28.085), "P": (15, 30.974),  "S": (16, 32.06),
    "Cl": (17, 35.45),  "Ar": (18, 39.948), "K": (19, 39.098),  "Ca": (20, 40.078),
    "Ti": (22, 47.867), "V": (23, 50.942),  "Cr": (24, 51.996), "Mn": (25, 54.938),
    "Fe": (26, 55.845), "Co": (27, 58.933), "Ni": (28, 58.693), "Cu": (29, 63.546),
    "Zn": (30, 65.38),  "Ge": (32, 72.630), "Br": (35, 79.904), "Kr": (36, 83.798),
    "Zr": (40, 91.224), "Nb": (41, 92.906), "Mo": (42, 95.95),  "Ag": (47, 107.87),
    "Sn": (50, 118.71), "I": (53, 126.90),  "Xe": (54, 131.29), "Ta": (73, 180.95),
    "W": (74, 183.84),  "Pt": (78, 195.08), "Au": (79, 196.97), "Pb": (82, 207.2),
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


@lru_cache(maxsize=256)
def parse_formula(formula: str) -> tuple[tuple[int, float, float], ...]:
    """
    "C4H10" -> ((1, 1.008, 10.0), (6, 12.011, 4.0))   i.e. (Z, A, atoms per formula unit)
    """
    text = formula.strip()
    if not text:
        raise ValueError("empty formula")

    counts: dict[str, float] = {}
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            break
        symbol, num = m.group(1), m.group(2)
        if symbol not in ELEMENTS:
            raise ValueError(f"formula '{formula}': unknown element '{symbol}'")
        n = float(num) if num else 1.0
        if n <= 0:
            raise ValueError(f"formula '{formula}': count of {symbol} must be positive")
        counts[symbol] = counts.get(symbol, 0.0) + n
        pos = m.end()
    if pos != len(text):
        raise ValueError(f"formula '{formula}' cannot be parsed (stopped at '{text[pos:]}')")

    return tuple((ELEMENTS[s][0], ELEMENTS[s][1], n)
                 for s, n in sorted(counts.items(), key=lambda kv: ELEMENTS[kv[0]][0]))
