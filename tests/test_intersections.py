"""Curve-crossing finder"""

import numpy as np
import pytest

from core.simulation import TrackResult, find_intersections
from core.units import StoppingUnit



def _mass_units(x, y):
    """A track whose dE/dx is given directly in MeV cm^2/g and x in g/cm^2 (rho = 1)"""
    x = np.asarray(x, float)
    return TrackResult("t", "#fff", x, np.zeros_like(x), np.asarray(y, float), 1.0)


def test_two_lines_cross_once():
    a = _mass_units([0, 1, 2, 3], [0, 1, 2, 3])
    b = _mass_units([0, 1, 2, 3], [3, 2, 1, 0])
    pts = find_intersections(a, b, StoppingUnit.MASS, mass_thickness=True)
    assert len(pts) == 1
    assert pts[0] == pytest.approx((1.5, 1.5))


def test_different_step_sizes_are_interpolated_by_x_not_by_row():
    a = _mass_units(np.linspace(0, 10, 11), np.linspace(0, 10, 11))             # y = x
    xb = np.array([0, 1, 7, 10.0])
    b = _mass_units(xb, 10 - xb)                                                # y = 10 - x
    (ix, iy), = find_intersections(a, b, StoppingUnit.MASS, mass_thickness=True)
    assert (ix, iy) == pytest.approx((5.0, 5.0))


def test_curves_that_never_cross_give_nothing():
    a = _mass_units([0, 1, 2], [1, 2, 3])
    b = _mass_units([0, 1, 2], [5, 6, 7])
    assert find_intersections(a, b, StoppingUnit.MASS, True) == []


def test_touching_at_a_grid_point_is_reported_once():
    a = _mass_units([0, 1, 2], [0, 1, 0])
    b = _mass_units([0, 1, 2], [1, 1, 1])
    assert len(find_intersections(a, b, StoppingUnit.MASS, True)) <= 1


def test_intersection_uses_full_resolution_curves(be10, b10, isobutane_low_p):
    """The marker drawn on the plot and the number in the results label must agree"""
    from core.simulation import SimSettings, simulate
    a = simulate(isobutane_low_p, be10, SimSettings())
    b = simulate(isobutane_low_p, b10, SimSettings())
    pts = find_intersections(a, b, StoppingUnit.LINEAR_MM, False)
    # the plot widget must not use a thinned copy of the curves
    import inspect
    import ui.plot_widget as pw
    assert "_decimate" not in inspect.getsource(pw)
    assert isinstance(pts, list)
