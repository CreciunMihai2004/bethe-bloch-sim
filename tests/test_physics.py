"""Stopping-power physics against published reference values"""

import numpy as np
import pytest

from core.physics import dEdx_components, dEdx_mass, zbl_nuclear_stopping
from core.particles import Particle
from core.materials import Material

# NIST PSTAR, liquid water, electronic stopping power (MeV cm^2 / g).
# https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
PSTAR_WATER = [
    # (E / MeV, S_el, tolerance)
    (100.0, 7.289, 0.01),
    (10.0, 45.67, 0.01),
    (1.0, 260.8, 0.04),     # no shell correction in the model -> a few % at 1 MeV
]


@pytest.mark.parametrize("E, ref, tol", PSTAR_WATER)
def test_proton_in_water_matches_pstar(water, proton, E, ref, tol):
    elec, _ = dEdx_components(water, proton, E)
    assert elec == pytest.approx(ref, rel=tol)


def test_stopping_is_positive_and_finite_over_the_whole_range(be10, isobutane_low_p):
    for E in np.logspace(-3, 1, 200):          # 1 keV .. 10 MeV
        s = dEdx_mass(isobutane_low_p, be10, E)
        assert np.isfinite(s) and s > 0, f"bad stopping power {s} at E={E} MeV"


def test_zero_energy_gives_zero_stopping(water, proton):
    assert dEdx_components(water, proton, 0.0) == (0.0, 0.0)


def test_nuclear_stopping_is_small_at_high_energy(water, proton):
    elec, nuc = dEdx_components(water, proton, 10.0)
    assert nuc < 1e-3 * elec


# ---- ZBL universal nuclear stopping ----

def test_zbl_reduced_stopping_peaks_at_epsilon_0327():
    """Published property of the ZBL universal curve (Hoessinger, Eq. 3.97 discussion)"""
    from core.physics import zbl_reduced_stopping
    eps = np.logspace(-3, 2, 200001)
    assert eps[np.argmax(zbl_reduced_stopping(eps))] == pytest.approx(0.327, abs=0.002)


def test_zbl_reduced_stopping_is_continuous_across_eps_30():
    from core.physics import zbl_reduced_stopping
    lo, hi = zbl_reduced_stopping(np.array([29.999, 30.001]))
    assert lo == pytest.approx(hi, rel=0.02)           # the published fit has a ~1 % step here


def test_zbl_absolute_stopping_matches_the_analytic_conversion():
    """S_n = pi a_U^2 gamma E / eps * s_n(eps)  with a_U = 0.8854 a0 / (Z1^.23 + Z2^.23)"""
    from core.physics import zbl_reduced_energy, zbl_reduced_stopping
    from core.constants import N_A
    be10 = Particle("Be10", z=4, M_u=10.0135)
    carbon = Material("C", Z=6, A=12.011, I_eV=81.0, rho=2.0)
    E_MeV = 0.1

    eps = zbl_reduced_energy(E_MeV * 1e3, be10.z, carbon.Z, be10.M_u, carbon.A)
    a0 = 0.529177e-8                                                # cm
    aU = 0.8854 * a0 / (be10.z ** 0.23 + carbon.Z ** 0.23)
    gamma = 4 * be10.M_u * carbon.A / (be10.M_u + carbon.A) ** 2
    S_atom_eV_cm2 = np.pi * aU ** 2 * gamma * (E_MeV * 1e6) / eps * zbl_reduced_stopping(eps)
    expected = S_atom_eV_cm2 * 1e-6 * N_A / carbon.A                # MeV cm^2 / g

    assert zbl_nuclear_stopping(E_MeV, be10, carbon) == pytest.approx(expected, rel=5e-3)


def test_zbl_nuclear_stopping_vanishes_at_zero_energy(water, proton):
    assert zbl_nuclear_stopping(0.0, proton, water) == 0.0


def test_array_input_matches_scalar_calls(be10, isobutane_low_p, water, proton):
    """dEdx_components is vectorized: one call on a grid == a call per energy"""
    E = np.geomspace(1e-5, 1e3, 400)          # includes energies at and below the validity limit
    for mat, part in ((isobutane_low_p, be10), (water, proton)):
        elec, nuc = dEdx_components(mat, part, E)
        one_by_one = np.array([dEdx_components(mat, part, float(e)) for e in E])
        assert elec == pytest.approx(one_by_one[:, 0], rel=1e-12, abs=0)
        assert nuc == pytest.approx(one_by_one[:, 1], rel=1e-12, abs=0)
        assert np.array_equal(elec == 0, one_by_one[:, 0] == 0)


def test_scalar_input_still_returns_plain_floats(water, proton):
    elec, nuc = dEdx_components(water, proton, 10.0)
    assert type(elec) is float and type(nuc) is float
    assert type(dEdx_mass(water, proton, 10.0)) is float


# ---- low-energy electronic stopping (velocity-proportional law matched to Bethe-Bloch) ----

def _log_slope(f, E, h=1e-3):
    return (np.log(f(E * (1 + h))) - np.log(f(E * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))


def test_electronic_stopping_goes_to_zero_like_sqrt_of_the_energy(be10, isobutane_low_p):
    from core.physics import _matching_point
    E_m, _ = _matching_point(isobutane_low_p, be10)
    elec = lambda E: dEdx_components(isobutane_low_p, be10, E)[0]
    for E in (1e-4, 1e-3, 1e-2, 0.5 * E_m):
        assert elec(E) > 0
        assert _log_slope(elec, E) == pytest.approx(0.5, abs=1e-6)
    assert elec(1e-4) < elec(1e-3) < elec(1e-2)


@pytest.mark.parametrize("mat_fixture, part_fixture", [("isobutane_low_p", "be10"),
                                                       ("isobutane_low_p", "b10"),
                                                       ("water", "proton")])
def test_low_energy_law_joins_bethe_bloch_smoothly(mat_fixture, part_fixture, request):
    """Same value and same slope on both sides of E_m (that is how E_m and the amplitude are fixed)"""
    from core.physics import _matching_point
    mat, part = request.getfixturevalue(mat_fixture), request.getfixturevalue(part_fixture)
    E_m, _ = _matching_point(mat, part)
    assert E_m > 0
    elec = lambda E: dEdx_components(mat, part, E)[0]
    assert elec(E_m * (1 - 1e-4)) == pytest.approx(elec(E_m * (1 + 1e-4)), rel=5e-4)
    # probe right next to the join: Bethe-Bloch itself bends away from slope 1/2 above E_m
    # (its log-slope changes by ~2 per unit of ln E for a proton in water), so 1 % away is too far
    assert _log_slope(elec, E_m * 0.9995, 2e-4) == pytest.approx(0.5, abs=0.01)
    assert _log_slope(elec, E_m * 1.0005, 2e-4) == pytest.approx(0.5, abs=0.01)


def test_high_energy_is_untouched_by_the_low_energy_law(water, proton):
    from core.physics import bethe_stopping
    for E in (1.0, 10.0, 100.0):
        assert dEdx_components(water, proton, E)[0] == bethe_stopping(water, proton, E)


@pytest.mark.parametrize("part_fixture, lo, hi", [("be10", 0.7, 1.05), ("b10", 0.8, 1.1)])
def test_low_energy_amplitude_agrees_with_lindhard_scharff(part_fixture, lo, hi, isobutane_low_p, request):
    """
    Independent check of the amplitude, which comes from Bethe-Bloch and has no free
    parameter: Lindhard-Scharff, S = 8 pi e^2 a0 * Z1^(7/6) Z2 / (Z1^(2/3) + Z2^(2/3))^(3/2) * v/v0,
    summed over the elements, must be within the usual few-10 % of it.
    """
    from core.constants import N_A
    part = request.getfixturevalue(part_fixture)
    E = np.array([0.01, 0.03, 0.1, 0.3])
    beta = np.sqrt(2.0 * E / part.M)
    ls = sum(n * 19.15 * part.z ** (7 / 6) * Z2 / (part.z ** (2 / 3) + Z2 ** (2 / 3)) ** 1.5
             for Z2, _, n in isobutane_low_p.composition) * beta * 137.036
    ls = ls * (N_A / isobutane_low_p.A) * 1e-21                      # eV/1e15 atoms/cm2 -> MeV cm2/g
    ratio = dEdx_components(isobutane_low_p, part, E)[0] / ls
    assert np.all((ratio > lo) & (ratio < hi)), ratio


def test_nuclear_stopping_is_summed_over_the_elements(be10):
    """C4H10 as C and H separately, not as one atom with Z = 34/14"""
    from core.physics import zbl_nuclear_stopping
    butane = Material("Butane", Z=34, A=58.124, I_eV=48.3, rho=1.0e-3, formula="C4H10")
    lumped = Material("Butane", Z=34, A=58.124, I_eV=48.3, rho=1.0e-3)
    E = 0.01
    by_element = zbl_nuclear_stopping(E, be10, butane)
    assert by_element != pytest.approx(zbl_nuclear_stopping(E, be10, lumped), rel=0.05)
    # reference: mass-weighted sum of the one-element stopping powers
    c = Material("C", Z=6, A=12.011, I_eV=81.0, rho=1.0)
    h = Material("H", Z=1, A=1.008, I_eV=19.2, rho=1.0)
    w_c, w_h = 4 * 12.011 / 58.124, 10 * 1.008 / 58.124
    assert by_element == pytest.approx(w_c * zbl_nuclear_stopping(E, be10, c)
                                       + w_h * zbl_nuclear_stopping(E, be10, h), rel=1e-9)
