"""Anode geometry: energy per anode and the pressure finder"""

from dataclasses import replace

import pytest

from core.beamline import Window, simulate_through_windows
from core.detector import AnodeLayout, anode_signals, crossings_mass, density_for_crossing
from core.materials import get_material
from core.particles import get_particle
from core.simulation import SimSettings, simulate


def _run(windows=()):
    return lambda mat, part: simulate_through_windows(mat, part, list(windows), SimSettings())


def test_layout_positions():
    a = AnodeLayout(gap_mm=5.0, anode1_mm=110.0, anode2_mm=210.0)
    assert (a.start_mm, a.boundary_mm, a.end_mm) == (5.0, 115.0, 325.0)
    with pytest.raises(ValueError):
        AnodeLayout(anode1_mm=-1.0)


def test_anode_energies_add_up_to_the_entry_energy(isobutane_low_p, be10):
    r = simulate(isobutane_low_p, be10, SimSettings())          # 385 mm: leaves anode 2
    s = anode_signals(r, AnodeLayout(gap_mm=10.0))
    assert s.gap + s.dE + s.E_res + s.escaped == pytest.approx(r.E[0], rel=1e-12)
    assert s.gap > 0 and s.dE > 0 and s.E_res > 0 and s.escaped > 0


def test_ion_stopping_inside_anode_2_leaves_nothing(isobutane_low_p, b10):
    r = simulate(isobutane_low_p, b10, SimSettings())           # 279 mm < 320 mm
    s = anode_signals(r, AnodeLayout())
    assert s.escaped == 0.0 and s.dE + s.E_res == pytest.approx(r.E[0], rel=1e-12)


def test_ion_stopping_before_the_boundary_has_no_residual_signal(isobutane_low_p, b10):
    r = simulate(isobutane_low_p, b10, SimSettings())
    s = anode_signals(r, AnodeLayout(anode1_mm=500.0))
    assert s.E_res == 0.0 and s.dE == pytest.approx(r.E[0])


@pytest.mark.parametrize("windows", [(), ("Mylar (C10H8O4)", 0.5)])
def test_density_puts_the_crossing_on_the_boundary(isobutane_low_p, be10, b10, windows):
    b10.E0 = 1.6
    wins = [Window(get_material(windows[0]), windows[1])] if windows else []
    target = AnodeLayout().boundary_mm
    rho = density_for_crossing(isobutane_low_p, [be10, b10], target, _run(wins))
    assert rho is not None and rho > 0
    gas = replace(isobutane_low_p, rho=rho)
    a, b = _run(wins)(gas, be10), _run(wins)(gas, b10)
    x_mm = [x / rho * 10.0 for x in crossings_mass(a, b)]
    assert min(abs(x - target) for x in x_mm) < 1e-3          # within 1 um


def test_density_finder_returns_none_without_a_crossing(isobutane_low_p, be10):
    other = replace(be10, E0=1.0)                               # same ion, lower energy: no crossing
    assert density_for_crossing(isobutane_low_p, [be10, other], 110.0, _run()) is None


def test_getters_return_copies():
    """Editing what get_material / get_particle return must not change the databases"""
    a = get_material("Isobutane (C4H10)")
    a.rho = 123.0
    assert get_material("Isobutane (C4H10)").rho != 123.0
    p = get_particle("Beryllium-10")
    p.E0 = 99.0
    assert get_particle("Beryllium-10").E0 != 99.0


def test_window_materials_are_separate_from_detector_media():
    from core.materials import MATERIAL_DB, WINDOW_MATERIAL_DB
    assert "Mylar (C10H8O4)" in WINDOW_MATERIAL_DB and "Mylar (C10H8O4)" not in MATERIAL_DB
    assert get_material("Mylar (C10H8O4)").rho == pytest.approx(1.40)
