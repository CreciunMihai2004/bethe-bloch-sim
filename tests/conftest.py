"""Shared test fixtures (headless Qt, common materials)"""

import os

# Qt needs a display; the offscreen platform lets the GUI tests run anywhere (CI included)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from core.materials import Material, get_material
from core.particles import Particle, get_particle


@pytest.fixture
def water() -> Material:
    """Liquid water, the NIST PSTAR reference medium (I = 75 eV)"""
    return Material("Water", Z=10, A=18.015, I_eV=75.0, rho=1.0)


@pytest.fixture
def proton() -> Particle:
    return Particle("Proton", z=1, M_u=1.00728)


@pytest.fixture
def isobutane_low_p() -> Material:
    """Isobutane at the 1.08e-5 g/cm3 used for the SRIM comparison (~4 mbar)"""
    mat = get_material("Isobutane (C4H10)")
    mat.rho = 1.08e-5
    return mat


@pytest.fixture
def be10() -> Particle:
    p = get_particle("Beryllium-10")
    p.E0 = 1.6
    return p


@pytest.fixture
def b10() -> Particle:
    p = get_particle("Boron-10")
    p.E0 = 1.4
    return p


_TEST_SETTINGS_ORG = "BraggSimTests"


@pytest.fixture(autouse=True)
def _isolated_session_settings(monkeypatch):
    """
    MainWindow.closeEvent saves the session, so every GUI test that closes a window would
    overwrite the user's real "Last Session". Point all tests at a separate settings name.
    """
    try:
        from ui.main_window import MainWindow
    except ImportError:                     # no PySide6: GUI tests are skipped anyway
        yield
        return
    monkeypatch.setattr(MainWindow, "_SETTINGS_ORG", _TEST_SETTINGS_ORG)
    yield
    from PySide6.QtCore import QSettings
    QSettings(_TEST_SETTINGS_ORG, MainWindow._SETTINGS_APP).clear()
