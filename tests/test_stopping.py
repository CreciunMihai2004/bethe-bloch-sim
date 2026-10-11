"""Tests for the stopping-power fixes (ZBL, smooth blend, track end)"""

import numpy as np
import pytest

from core import physics as ph
from core.elements import parse_formula
from core.materials import Material, get_material
from core.particles import Particle
from core.simulation import SimSettings, simulate
from core.units import StoppingUnit


@pytest.fixture(autouse=True)
def _blend_model():
    old = ph.get_default_model()
    ph.set_default_model("blend")
    yield
    ph.set_default_model(old)


@pytest.fixture
def isobutane():
    return Material(name="Isobutane (C4H10)", Z=34, A=58.124, I_eV=48.3, rho=0.00251)


@pytest.fixture
def be10():
    return Particle(name="Be-10", z=4, M_u=10.0135347, E0=4.0, color="#ff0000")


# ---- composition ----

def test_formula_parser():
    assert parse_formula("C4H10") == {"C": 4, "H": 10}
    assert parse_formula("He") == {"He": 1}
    assert parse_formula("Xx2") is None
    assert parse_formula("not a formula") is None


def test_isobutane_is_split_into_atoms(isobutane):
    assert isobutane.composition_source == "formula"
    assert sorted((Z, n) for Z, _, n in isobutane.components) == [(1, 10), (6, 4)]
    assert isobutane.mol_mass == pytest.approx(58.124, rel=0.02)


def test_elemental_material_is_single_atom():
    argon = Material(name="Argon (Ar)", Z=18, A=39.948, I_eV=188.0, rho=0.001662)
    assert argon.components == ((18, 39.948, 1),)


def test_formula_that_contradicts_csv_is_ignored():
    m = Material(name="Fake (CH4)", Z=34, A=58.124, I_eV=48.3, rho=0.0025)
    assert m.composition_source == "single-atom"


# ---- ZBL ----

def test_zbl_universal_function_is_continuous_at_eps_30():
    below, above = ph._zbl_reduced_stopping(29.999), ph._zbl_reduced_stopping(30.001)
    assert above == pytest.approx(below, rel=0.02)


def test_zbl_universal_function_positive_and_decreasing_at_high_eps():
    eps = np.geomspace(1.0, 1e4, 50)
    s = np.array([ph._zbl_reduced_stopping(e) for e in eps])
    assert np.all(s > 0)
    assert np.all(np.diff(s[eps > 30]) < 0)


def test_nuclear_stopping_decreases_with_energy(isobutane, be10):
    E = np.geomspace(0.01, 20, 40)
    sn = np.array([ph.zbl_nuclear_stopping(e, be10, isobutane) for e in E])
    assert np.all(sn > 0)
    assert np.all(np.diff(sn) < 0)


# ---- electronic blend ----

def test_electronic_is_positive_and_smooth(isobutane, be10):
    E = np.geomspace(1e-3, 50, 600)
    se = np.array([ph.dEdx_components(isobutane, be10, e)[0] for e in E])
    assert np.all(se > 0)                       # no hard cut-off to zero
    ratio = se[1:] / se[:-1]
    assert np.all((ratio > 0.9) & (ratio < 1.12))   # no jumps


def test_blend_matches_bethe_at_high_energy(isobutane, be10):
    E = 20.0
    assert ph.dEdx_components(isobutane, be10, E)[0] == pytest.approx(
        ph.bethe_stopping(isobutane, be10, E), rel=1e-3)


def test_blend_matches_lindhard_at_very_low_energy(isobutane, be10):
    E = 0.005
    assert ph.dEdx_components(isobutane, be10, E)[0] == pytest.approx(
        ph.lindhard_scharff_stopping(E, be10, isobutane), rel=1e-2)


def test_crossover_is_where_the_models_are_equal(isobutane, be10):
    Ex = ph.crossover_energy(isobutane, be10)
    assert Ex > 0
    assert ph.bethe_stopping(isobutane, be10, Ex) == pytest.approx(
        ph.lindhard_scharff_stopping(Ex, be10, isobutane), rel=1e-6)


# ---- track ----

def test_track_ends_at_zero_energy_and_has_one_peak(isobutane, be10):
    r = simulate(isobutane, be10, SimSettings())
    assert r.stopped
    assert r.E[-1] == 0.0
    assert np.all(np.diff(r.x_mass) > 0)

    # Main Bragg peak: dE/dx rises to it and then falls.  Checked down to
    # 50 keV only: below that the (real) nuclear-stopping bump of a heavy ion
    # makes the TOTAL dE/dx rise again slightly before it goes to zero
    m = np.isfinite(r.dEdx_mass)
    s, E = r.dEdx_mass[m], r.E[m]
    peak = int(np.argmax(s))
    assert np.all(np.diff(s[: peak + 1]) >= -1e-9 * s.max())
    after = s[peak:][E[peak:] > 0.05]
    assert np.all(np.diff(after) <= 1e-9 * s.max())


def test_range_equals_integral_of_inverse_stopping(isobutane, be10):
    r = simulate(isobutane, be10, SimSettings(max_dx=0.002, frac_loss=0.002))
    E = np.geomspace(1e-5, be10.E0, 40000)
    S = np.array([ph.dEdx_mass(isobutane, be10, e) for e in E])
    reference = np.sum(0.5 * (1 / S[1:] + 1 / S[:-1]) * np.diff(E))
    assert r.range_mass == pytest.approx(reference, rel=0.01)


def test_residual_range_is_small_and_positive(isobutane, be10):
    res = ph.residual_range_mass(isobutane, be10, 0.001)
    assert 0 < res < 1e-5          # g/cm2, tiny compared with the total range


def test_trim_tail_option_still_works(isobutane, be10):
    full = simulate(isobutane, be10, SimSettings(trim_tail=False))
    trimmed = simulate(isobutane, be10, SimSettings(trim_tail=True))
    assert len(trimmed.E) < len(full.E)
    assert full.range_mass > trimmed.range_mass


def test_track_leaving_the_medium_keeps_its_energy(isobutane, be10):
    r = simulate(isobutane, be10, SimSettings(x_stop=1.0))     # 1 mm: shorter than the range
    assert not r.stopped
    assert r.E[-1] > 0


# ---- SRIM table backend (tests/data holds a real SRIM export: Be-10 in n-butane) ----

from pathlib import Path                                   # noqa: E402
from core.srim_table import parse_srim_table               # noqa: E402

SRIM_FILE = Path(__file__).parent / "data" / "srim_be10_isobutane_1p6MeV.txt"


@pytest.fixture
def srim():
    t = parse_srim_table(SRIM_FILE)
    yield t
    ph.clear_srim_tables()


def test_srim_parser_reads_decimal_commas_and_units(srim):
    assert (srim.ion, srim.Z1, srim.mass_u) == ("Beryllium", 4, 10.0)
    assert srim.density == pytest.approx(1.08e-5)
    assert len(srim.E) == 59
    assert srim.E[0] == pytest.approx(0.010) and srim.E[-1] == pytest.approx(1.6)
    # SRIM gives MeV/(mg/cm2): 6.644 -> 6644 MeV cm2/g
    assert srim.Se[-1] == pytest.approx(6644.0)
    assert srim.Sn[0] == pytest.approx(605.1)
    assert srim.Rp_mm[-1] == pytest.approx(368.32)


def test_srim_model_reproduces_the_table(srim, isobutane, be10):
    ph.register_srim_table(be10, isobutane, srim)
    for E in (0.02, 0.1, 0.5, 1.0, 1.6):
        elec, nuc = ph.dEdx_components(isobutane, be10, E, "srim")
        assert elec == pytest.approx(srim.Se_at(E), rel=1e-9)
        # nuclear stays ZBL, which agrees with SRIM's nuclear column to ~1.5 %
        assert nuc == pytest.approx(srim.Sn_at(E), rel=0.02)


def test_srim_model_is_continuous_at_table_edges(srim, isobutane, be10):
    ph.register_srim_table(be10, isobutane, srim)
    for edge in (srim.E_min, srim.E_max):
        lo = ph.dEdx_components(isobutane, be10, edge * 0.9999, "srim")[0]
        hi = ph.dEdx_components(isobutane, be10, edge * 1.0001, "srim")[0]
        assert hi == pytest.approx(lo, rel=1e-3)


def test_srim_model_falls_back_without_a_table(isobutane, be10):
    ph.clear_srim_tables()
    assert ph.dEdx_components(isobutane, be10, 0.5, "srim") == ph.dEdx_components(isobutane, be10, 0.5, "blend")


def test_srim_range_matches_srim(srim, isobutane, be10):
    isobutane.rho = srim.density
    be10.E0 = 1.6
    ph.register_srim_table(be10, isobutane, srim)
    ph.set_default_model("srim")
    r = simulate(isobutane, be10, SimSettings(x_stop=5000))
    assert r.stopped and r.E[-1] == 0.0
    # CSDA path from the table (+ SRIM's own range below 10 keV)
    csda = srim.csda_mm() + srim.Rp_mm[0]
    assert r.range_mm == pytest.approx(csda, rel=0.01)
    # and 1-3 % above the projected range SRIM prints
    assert 1.00 < r.range_mm / srim.Rp_mm[-1] < 1.03


def test_blend_model_alone_is_known_to_be_off_vs_srim(srim, isobutane, be10):
    """Documents the limitation: the textbook Lindhard-Scharff term is ~3x low.
    If a better low-energy model is added this test should be tightened."""
    ratio = ph.dEdx_components(isobutane, be10, 0.02, "blend")[0] / srim.Se_at(0.02)
    assert 0.2 < ratio < 0.45
