"""
Benchmark against SRIM-2013 tables for the institute's detector gas
(n-butane / isobutane, 1.08e-5 g/cm3; Be-10 and B-10, 10 keV - 2 MeV).

SRIM is a benchmark, not a source: the app computes its own stopping powers. The tests
check the reader, pin down the parts that agree (nuclear stopping), and guard the size
of the known disagreement in the electronic stopping so it cannot get worse unnoticed.
See docs/PROGRESS.md for the numbers.
"""

from pathlib import Path

import numpy as np
import pytest

from core.physics import dEdx_components
from core.srim import read_srim_table

DATA = Path(__file__).parent / "data"
BE10 = DATA / "srim_be10_nbutane_1.08e-5.txt"
B10 = DATA / "srim_b10_nbutane_1.08e-5.txt"


def _path_length_mm(E, S, rho):
    """CSDA path length from E[0] up to every E (log-log interpolation of S, integral in ln E)"""
    lnE, lnS = np.log(E), np.log(S)
    fine = np.linspace(lnE[0], lnE[-1], 20000)
    f = np.exp(fine - np.interp(fine, lnE, lnS))                  # E / S
    x = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(fine))))
    return np.interp(lnE, fine, x) / rho * 10.0


# ---- reader ----

def test_reader_parses_a_romanian_locale_table():
    t = read_srim_table(BE10)
    assert (t.ion, t.ion_Z, t.ion_mass_u, t.density) == ("Beryllium", 4, 10.0, 1.08e-5)
    assert len(t.E) == 62
    assert t.E[0] == pytest.approx(0.010) and t.E[-1] == pytest.approx(2.0)
    i = int(np.argmin(abs(t.E - 1.6)))
    assert t.S_elec[i] == pytest.approx(6644.0)       # 6,644E+00 MeV/(mg/cm2) -> MeV cm2/g
    assert t.S_nuc[i] == pytest.approx(18.28)
    assert t.proj_range_mm[i] == pytest.approx(368.32)
    assert t.long_strag_mm[i] == pytest.approx(14.51)


def test_reader_accepts_decimal_points(tmp_path):
    import re
    text = re.sub(r"(?<=\d),(?=\d)", ".", BE10.read_text(encoding="latin-1"))
    p = tmp_path / "dot.txt"
    p.write_text(text, encoding="latin-1")
    a, b = read_srim_table(BE10), read_srim_table(p)
    assert np.array_equal(a.E, b.E) and np.array_equal(a.S_elec, b.S_elec)


def test_reader_rejects_other_files(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("not a SRIM table\n")
    with pytest.raises(ValueError):
        read_srim_table(p)


# ---- agreement ----

@pytest.mark.parametrize("path, part_fixture", [(BE10, "be10"), (B10, "b10")])
def test_nuclear_stopping_matches_srim(path, part_fixture, isobutane_low_p, request):
    """ZBL nuclear stopping, summed over C and H: within 1.5 % over the whole table"""
    t = read_srim_table(path)
    part = request.getfixturevalue(part_fixture)
    _, nuc = dEdx_components(isobutane_low_p, part, t.E)
    assert np.max(np.abs(nuc / t.S_nuc - 1.0)) < 0.015


@pytest.mark.parametrize("path, part_fixture, lo, hi", [
    (BE10, "be10", 1.00, 1.05),        # measured +2.6 %
    (B10, "b10", 1.10, 1.16),          # measured +12.8 %: electronic stopping too low below ~300 keV
])
def test_path_length_vs_srim_stopping_is_within_the_known_band(path, part_fixture, lo, hi,
                                                               isobutane_low_p, request):
    """
    Like-for-like: path length from 10 keV up to E0, from the app's stopping and from
    SRIM's own stopping columns (SRIM's projected range is a different quantity)
    """
    t = read_srim_table(path)
    part = request.getfixturevalue(part_fixture)
    m = t.E <= part.E0 + 1e-9
    E = t.E[m]
    elec, nuc = dEdx_components(isobutane_low_p, part, E)
    ratio = (_path_length_mm(E, elec + nuc, t.density)[-1]
             / _path_length_mm(E, t.S_total[m], t.density)[-1])
    assert lo < ratio < hi, ratio


def test_projected_range_is_close_to_path_length_at_detector_energies():
    """Why comparing with SRIM's projected range is fair at 1.4-1.6 MeV: they differ by ~1.5 %"""
    for path, E0 in ((BE10, 1.6), (B10, 1.4)):
        t = read_srim_table(path)
        i = int(np.argmin(abs(t.E - E0)))
        csda = t.proj_range_mm[0] + _path_length_mm(t.E, t.S_total, t.density)[i]
        assert 0.98 < t.proj_range_mm[i] / csda < 0.995
