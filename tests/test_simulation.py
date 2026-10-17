"""Integrator behaviour"""

import numpy as np
import pytest

from core.materials import Material
from core.physics import dEdx_mass
from core.simulation import SimSettings, simulate


def _csda_integral(mat, part, E_hi, E_lo, n=20001):
    """Independent range estimate R = integral dE / S(E) on a log-energy grid (g/cm^2)"""
    E = np.geomspace(E_lo, E_hi, n)
    f = 1.0 / dEdx_mass(mat, part, E)
    return float(np.trapezoid(f * E, np.log(E)))      # dE = E dlnE


def test_stepping_agrees_with_direct_integration(water, proton):
    proton.E0 = 10.0
    r = simulate(water, proton, SimSettings(x_stop=1e4))
    i = int(np.argmin(np.abs(r.E - 2.0)))             # compare the 10 MeV -> 2 MeV leg
    expected = _csda_integral(water, proton, 10.0, r.E[i])
    assert (r.x_mass[i] - r.x_mass[0]) == pytest.approx(expected, rel=2e-3)


def test_energy_decreases_monotonically(be10, isobutane_low_p):
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert np.all(np.diff(r.E) <= 0)
    assert np.all(np.diff(r.x_mass) > 0)


def test_array_lengths_are_consistent(be10, isobutane_low_p):
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert len(r.x_mass) == len(r.E) == len(r.dEdx_mass)


def test_missing_energy_is_rejected(water, proton):
    proton.E0 = None
    with pytest.raises(ValueError, match="initial kinetic energy"):
        simulate(water, proton, SimSettings())


def test_normal_run_ends_inside_the_model(water, proton):
    proton.E0 = 5.0
    r = simulate(water, proton, SimSettings(x_stop=1e4))
    assert r.stop_reason in ("stopped", "model_limit")
    assert r.is_complete


def test_stop_reason_when_depth_limit_cuts_the_track(water, proton):
    proton.E0 = 100.0
    r = simulate(water, proton, SimSettings(x_stop=10.0))          # 100 MeV needs ~77 mm
    assert r.stop_reason == "x_stop"


# ---- the track runs all the way down to rest ----

def test_heavy_ion_track_runs_down_to_zero_energy(be10, isobutane_low_p):
    """Used to stop with ~226 keV (14 %) unspent where Bethe-Bloch switched off"""
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert r.stop_reason == "stopped"
    assert r.E[-1] == 0.0 and r.unspent_fraction == 0.0
    assert r.dEdx_mass[-1] == 0.0
    assert np.all(np.isfinite(r.dEdx_mass))
    assert np.all(np.diff(r.E) < 0) and np.all(np.diff(r.x_mass) > 0)    # strictly down and forward


def test_the_low_energy_tail_is_a_small_part_of_the_range(be10, isobutane_low_p):
    """The analytic step from E_cutoff (0.1 keV) to rest must not carry the result"""
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert (r.x_mass[-1] - r.x_mass[-2]) / r.x_mass[-1] < 1e-3
    coarse = simulate(isobutane_low_p, be10, SimSettings(E_cutoff=1e-3)).range_mm
    assert coarse == pytest.approx(r.range_mm, rel=2e-3)      # cutting at 1 keV instead changes nothing visible


def test_range_regression_and_sanity_against_srim(be10, b10, isobutane_low_p):
    """
    Regression values of this model, plus a physical bound from the SRIM exports for the same
    cases (Be-10 1.6 MeV: 368.3 mm, B-10 1.4 MeV: 242.55 mm). SRIM reports the *projected*
    range, which is shorter than the path length this app integrates, so the model must not
    fall below it; it also must not drift far above it.
    """
    be = simulate(isobutane_low_p, be10, SimSettings()).range_mm
    b = simulate(isobutane_low_p, b10, SimSettings()).range_mm
    assert be == pytest.approx(385.07, rel=5e-3)
    assert b == pytest.approx(278.70, rel=5e-3)
    assert 368.3 <= be <= 1.10 * 368.3
    assert 242.55 <= b <= 1.20 * 242.55          # B-10 is the weaker case: see docs/PROGRESS.md


def test_starting_at_low_energy_gives_a_real_range(isobutane_low_p, be10):
    be10.E0 = 0.1                                    # below the old Bethe-Bloch validity limit
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert r.stop_reason == "stopped" and r.range_mm > 0.0
    be10.E0 = 1.6
    assert r.range_mm < simulate(isobutane_low_p, be10, SimSettings()).range_mm


def test_default_settings_have_no_depth_limit(be10):
    """The detector used to be capped at 1000 mm; a long track must now run to its end"""
    assert SimSettings().x_stop is None
    long_gas = Material("Thin gas", Z=34, A=58.124, I_eV=48.3, rho=1.0e-6, formula="C4H10")
    r = simulate(long_gas, be10, SimSettings())
    assert r.range_mm > 1000.0
    assert r.stop_reason == "stopped" and r.E[-1] == 0.0
    assert r.range_mm == pytest.approx(
        simulate(long_gas, be10, SimSettings(x_stop=1e9)).range_mm, rel=1e-12)


def test_an_explicit_depth_limit_still_works(be10):
    long_gas = Material("Thin gas", Z=34, A=58.124, I_eV=48.3, rho=1.0e-6, formula="C4H10")
    r = simulate(long_gas, be10, SimSettings(x_stop=1000.0))
    assert r.stop_reason == "x_stop" and r.range_mm == pytest.approx(1000.0, rel=1e-9)
    assert r.E[-1] > 0


def test_model_limit_fallback_when_there_is_no_low_energy_law(be10, isobutane_low_p, monkeypatch):
    """If Bethe-Bloch never reaches the matching slope the old behaviour (stop at the limit) remains"""
    import core.physics as physics
    from core.physics import dEdx_components
    monkeypatch.setattr(physics, "_matching_point", lambda mat, part: (0.0, 0.0))
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert r.stop_reason == "model_limit" and 0.10 < r.unspent_fraction < 0.20
    assert dEdx_components(isobutane_low_p, be10, r.E[-1])[0] > 0
    assert dEdx_components(isobutane_low_p, be10, r.E[-1] * (1 - 1e-4))[0] == 0
    be10.E0 = 0.1                                    # starts below the limit: empty track, no crash
    assert simulate(isobutane_low_p, be10, SimSettings()).range_mm == 0.0


def test_unspent_fraction_is_zero_for_a_particle_that_really_stops(water, proton):
    proton.E0 = 5.0
    r = simulate(water, proton, SimSettings(x_stop=1e4), )
    assert r.unspent_fraction < 0.01


# ---- grid integrator: accuracy and speed ----

def test_range_is_converged_in_the_energy_grid(be10, isobutane_low_p):
    """The default grid must sit on the converged value (a 25x finer grid changes nothing visible)"""
    coarse = simulate(isobutane_low_p, be10, SimSettings()).range_mm
    fine = simulate(isobutane_low_p, be10, SimSettings(dlnE=1e-4)).range_mm
    assert coarse == pytest.approx(fine, rel=5e-4)


def test_depth_limit_ends_the_track_exactly_at_x_stop(water, proton):
    proton.E0 = 100.0
    r = simulate(water, proton, SimSettings(x_stop=10.0))
    assert r.stop_reason == "x_stop"
    assert r.range_mm == pytest.approx(10.0, rel=1e-9)
    assert 0 < r.E[-1] < r.E[0]


def test_x_start_offsets_the_whole_track(water, proton):
    proton.E0 = 5.0
    a = simulate(water, proton, SimSettings(x_stop=1e4))
    b = simulate(water, proton, SimSettings(x_start=2.0, x_stop=1e4))
    assert b.x_mm[0] == pytest.approx(2.0)
    assert (b.x_mm - 2.0) == pytest.approx(a.x_mm)


def test_simulation_is_fast_enough_for_live_updates(be10, isobutane_low_p):
    """The old stepping loop needed ~330 ms for this case; the GUI re-runs it on every edit"""
    import time
    simulate(isobutane_low_p, be10, SimSettings())                 # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        simulate(isobutane_low_p, be10, SimSettings())
    assert (time.perf_counter() - t0) / 5 < 0.05
