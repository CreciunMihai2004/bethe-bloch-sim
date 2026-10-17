"""Windows in front of the detector: energy loss, stopping inside, units"""

from dataclasses import replace

import numpy as np
import pytest

from core.beamline import Window, pass_windows, simulate_through_windows
from core.materials import get_material
from core.physics import dEdx_mass
from core.simulation import SimSettings, simulate


@pytest.fixture
def mylar():
    return get_material("Mylar (C10H8O4)")


def test_no_windows_gives_the_plain_simulation(isobutane_low_p, be10):
    a = simulate(isobutane_low_p, be10, SimSettings())
    b = simulate_through_windows(isobutane_low_p, be10, [], SimSettings())
    assert np.array_equal(a.x_mass, b.x_mass) and np.array_equal(a.E, b.E)
    assert b.E_beam == pytest.approx(1.6) and b.window_loss == 0.0 and b.windows == []


def test_thin_window_loses_stopping_power_times_thickness(mylar, be10):
    """Very thin: dE = S(E0) * t to first order"""
    w = Window(mylar, 0.01)                                   # 10 nm
    p = pass_windows(be10, 1.6, [w])[0]
    assert p.loss == pytest.approx(dEdx_mass(mylar, be10, 1.6) * w.mass_thickness, rel=2e-3)


def test_two_halves_equal_one_window(mylar, be10):
    one = pass_windows(be10, 1.6, [Window(mylar, 1.0)])[-1].E_out
    two = pass_windows(be10, 1.6, [Window(mylar, 0.5), Window(mylar, 0.5)])[-1].E_out
    assert two == pytest.approx(one, rel=2e-3)


def test_energy_out_is_the_inverse_of_the_range(mylar, be10):
    """E_out is the energy whose residual range equals R(E_in) - t"""
    t_um = 1.5
    E_out = pass_windows(be10, 1.6, [Window(mylar, t_um)])[0].E_out
    R_in = simulate(mylar, be10, SimSettings()).range_mm
    R_out = simulate(mylar, replace(be10, E0=E_out), SimSettings()).range_mm
    assert R_in - R_out == pytest.approx(t_um * 1e-3, rel=5e-3)


def test_thickness_units_agree(mylar, be10):
    """1.5 um of Mylar (1.40 g/cm3) is 210 ug/cm2 = 0.21 mg/cm2 = 1500 nm"""
    windows = [Window(mylar, 1.5, "um"), Window(mylar, 210.0, "ug/cm2"),
               Window(mylar, 0.21, "mg/cm2"), Window(mylar, 1500.0, "nm")]
    assert {round(w.mass_thickness, 12) for w in windows} == {2.1e-4}
    outs = [pass_windows(be10, 1.6, [w])[0].E_out for w in windows]
    assert outs == pytest.approx([outs[0]] * 4, rel=1e-12)


def test_detector_track_starts_at_the_energy_after_the_windows(isobutane_low_p, mylar, be10):
    r = simulate_through_windows(isobutane_low_p, be10, [Window(mylar, 1.0)], SimSettings())
    assert r.E[0] == pytest.approx(r.windows[-1].E_out) and 0 < r.E[0] < 1.6
    assert r.window_loss == pytest.approx(1.6 - r.E[0])
    assert r.E[-1] == 0.0 and r.stop_reason == "stopped"
    assert r.range_mm < simulate(isobutane_low_p, be10, SimSettings()).range_mm


def test_particle_stopping_in_a_window(isobutane_low_p, mylar, be10):
    r = simulate_through_windows(isobutane_low_p, be10,
                                 [Window(mylar, 0.5), Window(mylar, 10.0), Window(mylar, 0.5)],
                                 SimSettings())
    assert r.stop_reason == "window" and r.is_complete
    assert len(r.windows) == 2 and r.windows[-1].stopped           # never reaches the third
    assert r.E[-1] == 0.0 and r.range_mm == 0.0 and r.window_loss == pytest.approx(1.6)


def test_zero_thickness_window_does_nothing(mylar, be10):
    assert pass_windows(be10, 1.6, [Window(mylar, 0.0)])[0].E_out == pytest.approx(1.6)


@pytest.mark.parametrize("kwargs", [dict(thickness=-1.0), dict(thickness=1.0, unit="inch"),
                                    dict(thickness=float("nan"))])
def test_bad_windows_are_rejected(mylar, kwargs):
    with pytest.raises(ValueError):
        Window(mylar, **kwargs)
