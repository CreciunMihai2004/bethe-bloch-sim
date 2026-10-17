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


# ---- mean excitation energy from the formula ----

@pytest.mark.parametrize("formula, gas, tabulated, rel", [
    ("CH4", True, 41.7, 2e-3),            # methane gas
    ("C4H10", True, 48.3, 1.5e-2),        # isobutane gas (ICRU 49 recommends 48.3 eV)
    ("C10H8O4", False, 78.7, 2e-3),       # Mylar
    ("C22H10N2O5", False, 79.6, 2e-3),    # Kapton
    ("C3H6", False, 56.5, 2e-3),          # polypropylene
    ("H2O", False, 75.0, 1e-2),           # liquid water
    ("NaI", False, 452.0, 2e-3),          # sodium iodide (1.13 x elemental rule)
])
def test_bragg_rule_reproduces_tabulated_compound_values(formula, gas, tabulated, rel):
    from core.elements import mean_excitation_energy
    assert mean_excitation_energy(parse_formula(formula), gas) == pytest.approx(tabulated, rel=rel)


def test_a_pure_element_keeps_its_own_value():
    from core.elements import ELEMENT_I_EV, mean_excitation_energy
    assert mean_excitation_energy(parse_formula("Ar"), gas=True) == ELEMENT_I_EV[18] == 188.0
    assert mean_excitation_energy(parse_formula("C"), gas=False) == 78.0


def test_every_parseable_element_has_an_I_value():
    from core.elements import ELEMENTS, ELEMENT_I_EV
    assert {z for z, _ in ELEMENTS.values()} <= set(ELEMENT_I_EV)


def test_material_takes_Z_A_and_I_from_its_formula():
    gas = Material("Isobutane", rho=2.51e-3, formula="C4H10")
    assert gas.Z == 34 and gas.A == pytest.approx(58.124) and gas.I_from_formula
    assert gas.I_eV == pytest.approx(47.85, abs=0.01)
    solid = Material("Isobutane, condensed", rho=0.56, formula="C4H10")
    assert solid.I_eV > gas.I_eV                  # bound C is 81 eV in a solid, 70 eV in a gas


def test_a_given_I_overrides_the_formula():
    m = Material("Isobutane", I_eV=48.3, rho=2.51e-3, formula="C4H10")
    assert m.I_eV == 48.3 and not m.I_from_formula


def test_density_change_keeps_the_mean_excitation_energy():
    from dataclasses import replace
    gas = Material("Isobutane", rho=2.51e-3, formula="C4H10")
    assert replace(gas, rho=1.08e-5).I_eV == gas.I_eV


def test_without_a_formula_everything_must_be_given():
    with pytest.raises(ValueError, match="I_eV"):
        Material("lump", Z=10, A=18.0, rho=1.0)
    with pytest.raises(ValueError, match="density"):
        Material("no density", formula="H2O")


def test_A_must_match_the_formula():
    with pytest.raises(ValueError, match="g/mol"):
        Material("bad", A=50.0, rho=1e-3, formula="C4H10")
