"""
Chemical elements and formula parsing (used for Bragg additivity)
"""

from __future__ import annotations

import math
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


# ---- mean excitation energies ----

# I of the pure elements, eV (ICRU 37 / NIST ESTAR); for the gases the value of the gas
ELEMENT_I_EV: dict[int, float] = {
    1: 19.2, 2: 41.8, 3: 40.0, 4: 63.7, 5: 76.0, 6: 78.0, 7: 82.0, 8: 95.0, 9: 115.0,
    10: 137.0, 11: 149.0, 12: 156.0, 13: 166.0, 14: 173.0, 15: 173.0, 16: 180.0, 17: 174.0,
    18: 188.0, 19: 190.0, 20: 191.0, 22: 233.0, 23: 245.0, 24: 257.0, 25: 272.0, 26: 286.0,
    27: 297.0, 28: 311.0, 29: 322.0, 30: 330.0, 32: 350.0, 35: 357.0, 36: 352.0, 40: 393.0,
    41: 417.0, 42: 424.0, 47: 470.0, 50: 488.0, 53: 491.0, 54: 482.0, 73: 718.0, 74: 727.0,
    78: 790.0, 79: 790.0, 82: 823.0,
}

# I of an atom BOUND in a compound (ICRU 37, Seltzer & Berger 1982): binding and phase change
# the light elements' values; other elements in a condensed compound take 1.13 x the element's
_BOUND_CONDENSED = {1: 19.2, 6: 81.0, 7: 82.0, 8: 106.0, 9: 112.0, 17: 180.0}
_BOUND_GAS = {1: 19.2, 6: 70.0}
_CONDENSED_FACTOR = 1.13


def mean_excitation_energy(composition: tuple[tuple[float, float, float], ...], gas: bool) -> float:
    """
    I (eV) of a compound from its elements by Bragg additivity, weighted by electrons:

        ln I = sum_i n_i Z_i ln I_i / sum_i n_i Z_i

    composition = ((Z, A, atoms per formula unit), ...) as parse_formula returns it. A single
    element gets its own tabulated I; in a compound each atom takes its bound value for the
    phase (gas: H 19.2, C 70 eV, others the element's own; condensed: H 19.2, C 81, N 82,
    O 106, F 112, Cl 180 eV, others 1.13 x the element's).
    Reproduces the tabulated values of methane, Mylar, Kapton and polypropylene exactly.
    """
    if not composition:
        raise ValueError("empty composition")
    if len(composition) == 1:
        Z = int(round(composition[0][0]))
        if Z not in ELEMENT_I_EV:
            raise ValueError(f"no mean excitation energy tabulated for Z = {Z}; give I_eV")
        return ELEMENT_I_EV[Z]

    weight = log_sum = 0.0
    for Z, _A, n in composition:
        Z = int(round(Z))
        if Z not in ELEMENT_I_EV:
            raise ValueError(f"no mean excitation energy tabulated for Z = {Z}; give I_eV")
        if gas:
            I = _BOUND_GAS.get(Z, ELEMENT_I_EV[Z])
        else:
            I = _BOUND_CONDENSED.get(Z, _CONDENSED_FACTOR * ELEMENT_I_EV[Z])
        weight += n * Z
        log_sum += n * Z * math.log(I)
    return math.exp(log_sum / weight)
