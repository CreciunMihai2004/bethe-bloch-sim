"""
Energy straggling: the rate, its propagation along a track, and the anode-signal widths.

The propagation is checked against a brute-force Monte Carlo (ions losing a Gaussian
random amount of energy in many small steps), which shares nothing with the analytic
formula except the rate and the stopping power.
"""

from dataclasses import replace

import numpy as np
import pytest

from core.beamline import Window, pass_windows, simulate_through_windows
from core.constants import K, ME_C2
from core.detector import AnodeLayout, anode_signals, separation
from core.materials import get_material
from core.physics import dEdx_mass
from core.simulation import SimSettings
from core.straggling import (energy_covariance, lindhard_scharff_factor, range_straggling_mass,
                             straggling_rate)

SIGMA_BEAM = 0.010          # MeV
DEPTHS = (0.0, 50.0, 110.0, 200.0, 320.0)   # mm: entrance, -, anode boundary, -, end of anode 2


@pytest.fixture(scope="module")
def monte_carlo():
    """Be-10, 1.6 MeV +- 10 keV, isobutane 1.08e-5 g/cm3: energies of 4000 ions at DEPTHS"""
    from core.materials import get_material
    from core.particles import get_particle
    gas = get_material("Isobutane (C4H10)")
    gas.rho = 1.08e-5
    be = get_particle("Beryllium-10")
    rng = np.random.default_rng(12345)
    N, dX = 4000, 5e-7                                     # g/cm^2 per step (0.46 mm)
    E = 1.6 + SIGMA_BEAM * rng.standard_normal(N)
    targets = [d * gas.rho / 10.0 for d in DEPTHS]
    snaps, X = {0.0: E.copy()}, 0.0
    while X < targets[-1]:
        live = E > 0
        S, w = dEdx_mass(gas, be, E[live]), straggling_rate(gas, be, E[live])
        step = S * dX - np.sqrt(w * dX) * rng.standard_normal(int(live.sum()))
        E[live] = np.maximum(E[live] - step, 0.0)
        X += dX
        for d, Xt in zip(DEPTHS, targets):
            if d not in snaps and X >= Xt - 1e-15:
                snaps[d] = E.copy()
    return gas, be, snaps


# ---- the rate ----

def test_fast_proton_gets_the_plain_bohr_rate(water, proton):
    """z_eff = 1 and all electrons take part: K m_e c^2 Z/A"""
    assert straggling_rate(water, proton, 100.0)[0] == pytest.approx(K * ME_C2 * 10 / 18.015, rel=1e-6)


def test_lindhard_scharff_factor_is_continuous_and_bounded():
    """0 at rest, rising like sqrt(chi), exactly 1 (Bohr) from chi ~ 2.2 on -- also at large chi,
    where the fit polynomial itself would turn over and go negative"""
    from core.straggling import _CHI_SWITCH
    assert 2.0 < _CHI_SWITCH < 2.5
    chi = np.linspace(0.0, 1000.0, 200001)
    f = lindhard_scharff_factor(chi)
    assert f[0] == 0.0 and np.all(f[chi >= _CHI_SWITCH] == 1.0)
    assert np.all(np.diff(f) >= -1e-12)                   # never decreases
    assert lindhard_scharff_factor(_CHI_SWITCH * (1 - 1e-9)) == pytest.approx(1.0, abs=1e-6)  # no jump


# ---- propagation ----

def test_energy_spread_matches_monte_carlo(monte_carlo):
    gas, be, snaps = monte_carlo
    r = simulate_through_windows(gas, replace(be, E0=1.6), [], SimSettings(), sigma_beam=SIGMA_BEAM)
    C = energy_covariance(r, DEPTHS[:-1])
    for i, d in enumerate(DEPTHS[:-1]):
        assert snaps[d].std() == pytest.approx(np.sqrt(C[i, i]), rel=0.05), d
    corr_mc = np.corrcoef(snaps[50.0], snaps[110.0])[0, 1]
    assert corr_mc == pytest.approx(C[1, 2] / np.sqrt(C[1, 1] * C[2, 2]), abs=0.02)


def test_anode_widths_match_monte_carlo(monte_carlo):
    gas, be, snaps = monte_carlo
    r = simulate_through_windows(gas, replace(be, E0=1.6), [], SimSettings(), sigma_beam=SIGMA_BEAM)
    s = anode_signals(r, AnodeLayout())
    dE, E_res = snaps[0.0] - snaps[110.0], snaps[110.0] - snaps[320.0]
    assert dE.std() == pytest.approx(s.sigma_dE, rel=0.06)
    assert E_res.std() == pytest.approx(s.sigma_E_res, rel=0.06)
    assert np.corrcoef(dE, E_res)[0, 1] == pytest.approx(s.corr, abs=0.03)


def test_range_spread_from_the_track_equals_the_direct_integral(isobutane_low_p, be10):
    r = simulate_through_windows(isobutane_low_p, be10, [], SimSettings(), sigma_beam=SIGMA_BEAM)
    direct = range_straggling_mass(isobutane_low_p, be10, 1.6, SIGMA_BEAM ** 2)
    assert r.range_sigma_mm == pytest.approx(direct / isobutane_low_p.rho * 10.0, rel=0.01)


def test_beam_spread_alone_scales_with_the_stopping_power(isobutane_low_p, be10, monkeypatch):
    """No straggling: an ion delta above the mean stays delta/S ahead, so sigma(x) = sigma0 S(x)/S0"""
    import core.straggling as st
    monkeypatch.setattr(st, "straggling_rate", lambda mat, part, E: np.zeros_like(np.atleast_1d(E)))
    r = simulate_through_windows(isobutane_low_p, be10, [], SimSettings(), sigma_beam=SIGMA_BEAM)
    ok = r.dEdx_mass > 0
    expected = SIGMA_BEAM * r.dEdx_mass[ok] / r.dEdx_mass[0]
    assert r.sigma_E[ok] == pytest.approx(expected, rel=1e-9)


def test_straggling_off_gives_no_widths(isobutane_low_p, be10):
    r = simulate_through_windows(isobutane_low_p, be10, [], SimSettings(), straggling=False)
    assert r.strag_J is None and r.sigma_E is None and r.range_sigma_mm is None
    s = anode_signals(r, AnodeLayout())
    assert s.sigma_dE is None and s.covariance is None


def test_windows_add_spread_and_pass_it_on(be10):
    mylar = get_material("Mylar (C10H8O4)")
    thin = pass_windows(be10, 1.6, [Window(mylar, 0.2)])[0].sigma_out
    thick = pass_windows(be10, 1.6, [Window(mylar, 1.0)])[0].sigma_out
    assert 0 < thin < thick
    two = pass_windows(be10, 1.6, [Window(mylar, 0.5), Window(mylar, 0.5)])[-1].sigma_out
    assert two == pytest.approx(thick, rel=0.01)                      # same as one 1 um window


def test_detector_starts_with_the_spread_out_of_the_windows(isobutane_low_p, be10):
    wins = [Window(get_material("Mylar (C10H8O4)"), 0.5)]
    r = simulate_through_windows(isobutane_low_p, be10, wins, SimSettings(), sigma_beam=SIGMA_BEAM)
    assert r.sigma_E[0] == pytest.approx(r.windows[-1].sigma_out, rel=1e-9)
    assert r.windows[-1].sigma_out > SIGMA_BEAM * 0.5


# ---- anode signals ----

def test_signal_covariance_is_a_valid_covariance(isobutane_low_p, be10, b10):
    for part in (be10, b10):
        r = simulate_through_windows(isobutane_low_p, part, [], SimSettings(), sigma_beam=SIGMA_BEAM)
        C = anode_signals(r, AnodeLayout()).covariance
        assert np.all(np.linalg.eigvalsh(C) >= -1e-15)


def test_ion_stopping_in_anode_2_deposits_all_that_reaches_it(isobutane_low_p, b10):
    """E_res = E(boundary) exactly, so its width is the energy spread at the boundary"""
    r = simulate_through_windows(isobutane_low_p, b10, [], SimSettings(), sigma_beam=SIGMA_BEAM)
    layout = AnodeLayout()
    s = anode_signals(r, layout)
    assert s.escaped == 0.0
    C = energy_covariance(r, [layout.boundary_mm])
    assert s.sigma_E_res == pytest.approx(np.sqrt(C[0, 0]), rel=1e-9)


def test_separation(isobutane_low_p, be10, b10):
    layout = AnodeLayout()
    b10.E0 = 1.6
    a = anode_signals(simulate_through_windows(isobutane_low_p, be10, [], SimSettings()), layout)
    b = anode_signals(simulate_through_windows(isobutane_low_p, b10, [], SimSettings()), layout)
    assert separation(a, a) == 0.0
    d = separation(a, b)
    assert d > 2.0                                             # Be and B are far apart here
    assert separation(a, replace(b, dE=a.dE + 0.5 * (b.dE - a.dE),
                                 E_res=a.E_res + 0.5 * (b.E_res - a.E_res))) < d
