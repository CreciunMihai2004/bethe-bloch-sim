"""Integrator behaviour"""

import numpy as np
import pytest

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


def test_progress_callback_reaches_one(water, proton):
    proton.E0 = 5.0
    seen = []
    simulate(water, proton, SimSettings(), seen.append)
    assert seen and seen[-1] == 1.0


def test_normal_run_ends_inside_the_model(water, proton):
    proton.E0 = 5.0
    r = simulate(water, proton, SimSettings(x_stop=1e4))
    assert r.stop_reason in ("stopped", "model_limit")
    assert r.is_complete


def test_stop_reason_when_depth_limit_cuts_the_track(water, proton):
    proton.E0 = 100.0
    r = simulate(water, proton, SimSettings(x_stop=10.0))          # 100 MeV needs ~77 mm
    assert r.stop_reason == "x_stop"


# ---- the electronic-stopping model limit is explicit, not trimmed away ----

def test_heavy_ion_track_ends_at_the_model_limit_and_says_so(be10, isobutane_low_p):
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert r.stop_reason == "model_limit"
    assert 0.10 < r.unspent_fraction < 0.20          # ~226 keV of 1.6 MeV is not tracked
    assert np.all(np.isfinite(r.dEdx_mass))          # no NaN left at the end of the arrays


def test_range_is_unchanged_by_replacing_the_trim_heuristic(be10, b10, isobutane_low_p):
    """Regression guard: the old 1.5 %-of-peak trim gave 372.71 mm / 273.41 mm"""
    assert simulate(isobutane_low_p, be10, SimSettings()).range_mm == pytest.approx(372.71, rel=5e-3)
    assert simulate(isobutane_low_p, b10, SimSettings()).range_mm == pytest.approx(273.41, rel=5e-3)


def test_starting_below_the_model_limit_does_not_crash(isobutane_low_p, be10):
    be10.E0 = 0.1                                    # below the Bethe-Bloch validity limit
    r = simulate(isobutane_low_p, be10, SimSettings())
    assert r.range_mm == 0.0 and r.stop_reason == "model_limit"


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


def test_model_limit_track_ends_exactly_at_the_limit(be10, isobutane_low_p):
    """Stopping power is small there, so ending a grid step early would drop part of the range"""
    from core.physics import dEdx_components
    r = simulate(isobutane_low_p, be10, SimSettings())
    E_end = r.E[-1]
    assert dEdx_components(isobutane_low_p, be10, E_end)[0] > 0
    assert dEdx_components(isobutane_low_p, be10, E_end * (1 - 1e-4))[0] == 0


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
