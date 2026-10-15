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
