"""Chemical formula parsing and the formula/Z consistency check"""

import pytest

from core.elements import parse_formula
from core.materials import Material, get_material


def test_parse_isobutane():
    assert parse_formula("C4H10") == ((1, 1.008, 10.0), (6, 12.011, 4.0))      # ordered by Z


def test_parse_single_element_and_implicit_counts():
    assert parse_formula("Ar") == ((18, 39.948, 1.0),)
    assert [z for z, _, _ in parse_formula("CH4")] == [1, 6]
    assert dict((z, n) for z, _, n in parse_formula("CH4")) == {1: 4.0, 6: 1.0}


def test_repeated_elements_are_merged():
    assert dict((z, n) for z, _, n in parse_formula("CH3CH3")) == {1: 6.0, 6: 2.0}


@pytest.mark.parametrize("bad", ["", "  ", "Xx2", "c4h10", "C4H10)", "(CH3)2", "C0H2", "4C"])
def test_bad_formulas_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_formula_must_match_the_electron_count():
    Material("ok", Z=34, A=58.124, I_eV=48.3, rho=1e-3, formula="C4H10")
    with pytest.raises(ValueError, match="electrons"):
        Material("bad", Z=30, A=58.124, I_eV=48.3, rho=1e-3, formula="C4H10")


def test_no_formula_means_one_effective_atom():
    m = Material("lump", Z=10, A=18.015, I_eV=75.0, rho=1.0)
    assert m.composition == ((10.0, 18.015, 1.0),)


def test_every_shipped_material_has_a_consistent_formula():
    from core.materials import MATERIAL_DB, WINDOW_MATERIAL_DB
    for name in list(MATERIAL_DB) + list(WINDOW_MATERIAL_DB):
        m = get_material(name)
        assert m.formula, f"{name}: add a formula to materials.csv"
        z_sum = sum(z * n for z, _, n in m.composition)
        a_sum = sum(a * n for _, a, n in m.composition)
        assert z_sum == pytest.approx(m.Z, abs=0.5)
        assert a_sum == pytest.approx(m.A, rel=2e-3), f"{name}: A {m.A} vs formula {a_sum:.3f}"
