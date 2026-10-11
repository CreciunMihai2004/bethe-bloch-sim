"""
Stopping power

    electronic : Lindhard-Scharff (low energy)  ->  Bethe-Bloch (high energy),
                 joined by a smooth weight in ln(E/u)
    nuclear    : ZBL universal screening, evaluated per target atom

Three models can be selected (see set_default_model):

    "blend"  : the model described above (default)
    "srim"   : electronic stopping taken from a SRIM table registered for the
               ion/material pair (register_srim_table); nuclear stays ZBL.
               Falls back to "blend" for any pair without a table
    "legacy" : the original behaviour, kept only so the diagnostic script can
               show old-vs-new side by side

KNOWN LIMITATION of "blend": the low-energy (Lindhard-Scharff) term is the
textbook coefficient, which for Be-10 in isobutane is about 3x BELOW SRIM
between 10 and 100 keV (see tools/diagnose_stopping.py with a SRIM table).
A single scale factor cannot repair that; use "srim" tables until a properly
fitted low-energy model replaces it.
"""

import numpy as np

from .constants import K, ME_C2, N_A
from .materials import Material
from .particles import Particle
from .srim_table import SrimTable

# ---- model selection ----

MODEL_BLEND = "blend"
MODEL_LEGACY = "legacy"
MODEL_SRIM = "srim"
_MODELS = (MODEL_BLEND, MODEL_SRIM, MODEL_LEGACY)
_default_model = MODEL_BLEND


def set_default_model(model: str) -> None:
    global _default_model
    if model not in _MODELS:
        raise ValueError(f"Unknown stopping model '{model}'. Choose from {_MODELS}")
    _default_model = model


def get_default_model() -> str:
    return _default_model


def _resolve(model):
    if model is None:
        return _default_model
    if model not in _MODELS:
        raise ValueError(f"Unknown stopping model '{model}'. Choose from {_MODELS}")
    return model


# ---- tunable parameters (calibration knobs) ----

# Lindhard-Scharff / Bethe hand-over.  The two curves are joined with a smooth
# weight centred on the energy E_x where they are EQUAL (found numerically per
# ion/material), so the hand-over has no jump whatever the ion.  Above E_x the
# result is Bethe, below it Lindhard-Scharff.  BLEND_WIDTH is the width of the
# transition in ln(E); smaller = sharper (the curves are equal at E_x, so even
# a sharp hand-over is continuous).
BLEND_WIDTH = 0.3
# Multiplies the Lindhard-Scharff term (1.0 = textbook coefficient).  Changing
# it moves E_x, so it is the main knob for matching SRIM at low energy.
LOW_E_SCALE = 1.0

_E_FLOOR = 1e-6     # MeV; below this (1 eV) all stopping is treated as zero


# ---- small helpers ----

def _atoms_to_mass(sum_n_S: float, mat: Material) -> float:
    """
    sum_n_S : sum over the atoms of one molecule of n_i * S_i,
              S_i in eV / (1e15 atoms/cm^2)
    returns mass stopping power in MeV cm^2 / g
    """
    return sum_n_S * N_A / mat.mol_mass * 1e-21


def density_effect(beta: float, gamma: float, mat: Material) -> float:
    """Sternheimer density-effect correction delta(X), X = log10(beta*gamma)"""
    bg = beta * gamma
    if bg <= 0:
        return 0.0
    X = np.log10(bg)
    if X < mat.X0:
        return 0.0
    elif X < mat.X1:
        return 4.6052 * X - mat.Cbar + mat.a_stern * (mat.X1 - X) ** mat.m_stern
    else:
        return 4.6052 * X - mat.Cbar


def effective_charge(z: int, beta: float) -> float:
    """
    Effective charge
    """
    if beta <= 0:
        return 0.0
    return z * (1.0 - np.exp(-125.0 * beta * z ** (-2.0 / 3.0)))


def _kinematics(part: Particle, E_kin: float):
    """(beta, gamma, beta^2) from kinetic energy; stable at very small beta"""
    x = E_kin / part.M
    gamma = 1.0 + x
    beta_sq = max(x * (x + 2.0) / gamma ** 2, 1e-12)
    return np.sqrt(beta_sq), gamma, beta_sq


# ---- nuclear stopping (ZBL) ----

def _zbl_reduced_stopping(eps: float) -> float:
    """ZBL universal reduced nuclear stopping s_n(epsilon)"""
    if eps <= 0.0:
        return 0.0
    if eps <= 30.0:
        return np.log(1.0 + 1.1383 * eps) / (
            2.0 * (eps + 0.01321 * eps ** 0.21226 + 0.19593 * eps ** 0.5)
        )
    return np.log(eps) / (2.0 * eps)


def _zbl_atom(E_keV: float, Z1: float, M1: float, Z2: float, M2: float) -> float:
    """ZBL nuclear stopping cross-section, eV / (1e15 atoms/cm^2), one target atom"""
    screen = Z1 ** 0.23 + Z2 ** 0.23
    eps = 32.53 * M2 * E_keV / (Z1 * Z2 * (M1 + M2) * screen)
    s_n = _zbl_reduced_stopping(eps)
    return 8.462 * Z1 * Z2 * M1 * s_n / ((M1 + M2) * screen)


def _zbl_legacy(E_kin: float, part: Particle, mat: Material) -> float:
    """Original implementation, kept for comparison only"""
    E_keV = E_kin * 1000.0
    Z1, M1, Z2, M2 = part.z, part.M_u, mat.Z, mat.A
    denominator = Z1 * Z2 * (M1 + M2) * (Z1 ** 0.67 + Z2 ** 0.67) ** 0.5
    epsilon = (32.53 * M2 * E_keV) / denominator
    num = 0.5 * np.log(1.0 + epsilon)
    den = epsilon + 0.01321 * (epsilon ** 0.21226) + 0.19593 * (epsilon ** 0.5)
    s_n = num / den
    S_n = (8.462 * Z1 * Z2 * M1 * s_n) / ((M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))
    return max(S_n * (N_A / M2) * 1e-21, 0.0)


def zbl_nuclear_stopping(E_kin: float, part: Particle, mat: Material,
                         model: str | None = None) -> float:
    """
    Nuclear mass stopping power (MeV cm^2 / g), ZBL universal formula,
    summed over the atoms of the target molecule (Bragg additivity)
    """
    model = _resolve(model)
    if E_kin <= _E_FLOOR:
        return 0.0
    if model == MODEL_LEGACY:
        return _zbl_legacy(E_kin, part, mat)

    E_keV = E_kin * 1000.0
    total = sum(n * _zbl_atom(E_keV, part.z, part.M_u, Z2, M2)
                for Z2, M2, n in mat.components)
    return max(_atoms_to_mass(total, mat), 0.0)


# ---- electronic stopping ----

def lindhard_scharff_stopping(E_kin: float, part: Particle, mat: Material) -> float:
    """
    Low-energy electronic stopping, S_e ~ sqrt(E) (Lindhard-Scharff form as
    given in the ZBL book), summed over target atoms.  MeV cm^2 / g
    """
    if E_kin <= _E_FLOOR:
        return 0.0
    Z1, M1 = part.z, part.M_u
    root_E = np.sqrt(E_kin * 1000.0)            # sqrt(keV)

    total = 0.0
    for Z2, _M2, n in mat.components:
        k = 1.212 * Z1 ** (7.0 / 6.0) * Z2 / (
            (Z1 ** (2.0 / 3.0) + Z2 ** (2.0 / 3.0)) ** 1.5 * np.sqrt(M1)
        )
        total += n * k * root_E
    return LOW_E_SCALE * _atoms_to_mass(total, mat)


def bethe_stopping(mat: Material, part: Particle, E_kin: float) -> float:
    """
    Bethe-Bloch electronic stopping with effective charge and density effect,
    in the form ln(1 + x) instead of ln(x) so that it stays positive and finite
    at low energy (identical to ln(x) to better than 1e-3 above ~0.5 MeV/u).
    Returns 0 if the effective charge is 0.  MeV cm^2 / g
    """
    if E_kin <= _E_FLOOR:
        return 0.0
    beta, gamma, beta_sq = _kinematics(part, E_kin)
    z_eff = effective_charge(part.z, beta)
    if z_eff <= 0:
        return 0.0

    ratio = ME_C2 / part.M
    T_max = (2.0 * ME_C2 * beta_sq * gamma ** 2) / (
        1.0 + 2.0 * gamma * ratio + ratio ** 2
    )
    log_arg = (2.0 * ME_C2 * beta_sq * gamma ** 2 * T_max) / (mat.I_MeV ** 2)
    delta = density_effect(beta, gamma, mat)

    s = K * z_eff ** 2 * (mat.Z / mat.A) / beta_sq * (
        0.5 * np.log1p(log_arg) - beta_sq - delta / 2.0
    )
    return max(s, 0.0)


_crossover_cache: dict = {}


def crossover_energy(mat: Material, part: Particle) -> float:
    """
    Lowest energy (MeV) at which Bethe-Bloch stopping rises above the
    Lindhard-Scharff stopping, i.e. where the two models are equal.
    Returns 0.0 if Bethe is above Lindhard-Scharff over the whole scan.
    Cached: depends only on the ion and the material's composition and I.
    """
    key = (part.z, part.M_u, mat.Z, mat.A, mat.I_eV, mat.components, LOW_E_SCALE)
    if key in _crossover_cache:
        return _crossover_cache[key]

    def gap(E):      # > 0 where Bethe is above Lindhard-Scharff
        return bethe_stopping(mat, part, E) - lindhard_scharff_stopping(E, part, mat)

    grid = np.geomspace(1e-4 * part.M_u, 1.0 * part.M_u, 300)    # 0.1 keV/u .. 1 MeV/u
    vals = np.array([gap(E) for E in grid])
    above = np.where(vals >= 0)[0]
    if len(above) == 0:
        E_x = float(grid[-1])
    elif above[0] == 0:
        E_x = 0.0
    else:
        lo, hi = grid[above[0] - 1], grid[above[0]]
        for _ in range(60):                                      # bisection in ln(E)
            mid = np.sqrt(lo * hi)
            if gap(mid) >= 0:
                hi = mid
            else:
                lo = mid
        E_x = float(np.sqrt(lo * hi))

    _crossover_cache[key] = E_x
    return E_x


def _blend_weight_low(E_kin: float, E_x: float) -> float:
    """Weight of the low-energy (Lindhard-Scharff) term: 1 at low E, 0 at high E"""
    if E_x <= 0.0:
        return 0.0
    x = np.log(E_kin / E_x) / BLEND_WIDTH
    x = min(max(x, -50.0), 50.0)
    return 1.0 / (1.0 + np.exp(x))


def _electronic_blend(mat: Material, part: Particle, E_kin: float) -> float:
    s_low = lindhard_scharff_stopping(E_kin, part, mat)
    s_high = bethe_stopping(mat, part, E_kin)
    w = _blend_weight_low(E_kin, crossover_energy(mat, part))
    return w * s_low + (1.0 - w) * s_high


def _electronic_legacy(mat: Material, part: Particle, E_kin: float) -> float:
    """Original behaviour: plain Bethe, forced to zero when the log argument is small"""
    beta, gamma, beta_sq = _kinematics(part, E_kin)
    z_eff = effective_charge(part.z, beta)
    if z_eff <= 0:
        return 0.0
    ratio = ME_C2 / part.M
    T_max = (2.0 * ME_C2 * beta_sq * gamma ** 2) / (
        1.0 + 2.0 * gamma * ratio + ratio ** 2
    )
    log_arg = (2.0 * ME_C2 * beta_sq * gamma ** 2 * T_max) / (mat.I_MeV ** 2)
    if log_arg > 1.05:
        delta = density_effect(beta, gamma, mat)
        bb = K * z_eff ** 2 * (mat.Z / mat.A) / beta_sq * (
            0.5 * np.log(log_arg) - beta_sq - delta / 2.0
        )
        return max(bb, 0.0)
    return 0.0


# ---- SRIM table registry ----

class _TableEntry:
    def __init__(self, table: SrimTable):
        self.table = table
        self.top_scale = None       # blend -> table factor at the top of the table


_SRIM_TABLES: dict = {}


def register_srim_table(part: Particle, mat: Material, table: SrimTable) -> None:
    """Use `table` for the electronic stopping of `part` in `mat` (model "srim")"""
    _SRIM_TABLES[(part.name, mat.name)] = _TableEntry(table)


def clear_srim_tables() -> None:
    _SRIM_TABLES.clear()


def has_srim_table(part: Particle, mat: Material) -> bool:
    return (part.name, mat.name) in _SRIM_TABLES


def _electronic_srim(entry: _TableEntry, mat: Material, part: Particle, E_kin: float) -> float:
    """
    Inside the table: log-log interpolation.
    Below it: Se(E_min) * sqrt(E / E_min)  (the velocity-proportional regime).
    Above it: the blend model scaled to join the table value at E_max, so
              the curve stays continuous - export the table up to the highest
              energy you need to avoid relying on this
    """
    t = entry.table
    if E_kin < t.E_min:
        return t.Se[0] * np.sqrt(E_kin / t.E_min)
    if E_kin <= t.E_max:
        return t.Se_at(E_kin)
    if entry.top_scale is None:
        ref = _electronic_blend(mat, part, t.E_max)
        entry.top_scale = t.Se[-1] / ref if ref > 0 else 1.0
    return entry.top_scale * _electronic_blend(mat, part, E_kin)


# ---- public API ----

def dEdx_components(mat: Material, part: Particle, E_kin: float,
                    model: str | None = None) -> tuple[float, float]:
    """
    Mass stopping power split into its two parts (MeV cm^2 / g):

        (electronic, nuclear)
    """
    model = _resolve(model)
    if E_kin <= _E_FLOOR:
        return 0.0, 0.0

    nuc = zbl_nuclear_stopping(E_kin, part, mat, model)
    if model == MODEL_LEGACY:
        elec = _electronic_legacy(mat, part, E_kin)
    elif model == MODEL_SRIM and has_srim_table(part, mat):
        elec = _electronic_srim(_SRIM_TABLES[(part.name, mat.name)], mat, part, E_kin)
    else:
        elec = _electronic_blend(mat, part, E_kin)
    return elec, nuc


def dEdx_mass(mat: Material, part: Particle, E_kin: float,
              model: str | None = None) -> float:
    """
    TOTAL MASS stopping power (Electronic + Nuclear)
    """
    elec, nuc = dEdx_components(mat, part, E_kin, model)
    return elec + nuc


def residual_range_mass(mat: Material, part: Particle, E_kin: float,
                        model: str | None = None, n_points: int = 400) -> float:
    """
    CSDA path left for a particle of kinetic energy E_kin (g/cm^2):

        R(E) = integral_0^E dE' / S(E')

    Substituting E' = u^2 turns the integrable 1/sqrt(E) behaviour at the end
    of the track into a smooth integrand 2u / S(u^2).  Below the 1 eV floor the
    contribution is negligible and ignored.
    """
    if E_kin <= _E_FLOOR:
        return 0.0
    u = np.linspace(np.sqrt(_E_FLOOR), np.sqrt(E_kin), n_points)
    S = np.array([dEdx_mass(mat, part, e * e, model) for e in u])
    ok = S > 0
    if ok.sum() < 2:
        return 0.0
    f, uu = 2.0 * u[ok] / S[ok], u[ok]
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(uu)))
