"""The Qt window builds, simulates and draws"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from core.simulation import SimSettings
from core.pipeline import SimJob, run_job


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_run_job_gives_one_track_per_particle(water, proton):
    proton.E0 = 5.0
    got = run_job(SimJob(water, [proton, proton]))
    assert len(got) == 2 and got[0].strag_J is not None


def test_main_window_draws_results(qapp, water, proton):
    from ui.main_window import MainWindow
    win = MainWindow()
    proton.E0 = 5.0
    job = SimJob(water, [proton], SimSettings())
    win.on_sim_done(run_job(job))
    assert win._results
    assert "Range (mm)" in win.results.text()
    win.close()


def test_edited_particle_is_labelled_custom(qapp):
    from ui.main_window import ParticlePanel
    panel = ParticlePanel("P", "Beryllium-10")
    assert panel.particle().name == "Beryllium-10"
    panel.spin_z.setValue(panel.spin_z.value() + 1)
    assert panel.particle().name == "Beryllium-10 (custom)"


def test_results_warns_when_track_is_cut(qapp, water, proton):
    from ui.main_window import MainWindow
    win = MainWindow()
    proton.E0 = 100.0
    win.on_sim_done(run_job(SimJob(water, [proton], SimSettings(x_stop=10.0))))
    assert "cut by the depth limit" in win.results.text()
    win.close()


def test_results_has_no_warning_for_a_track_that_runs_to_rest(qapp, water, proton, be10, isobutane_low_p):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.detector_panel.box.setChecked(False)          # no anodes: nothing to say about E_res

    proton.E0 = 5.0
    win.on_sim_done(run_job(SimJob(water, [proton], SimSettings())))
    assert "⚠" not in win.results.text()

    win.on_sim_done(run_job(SimJob(isobutane_low_p, [be10], SimSettings())))   # used to warn: ~14 % unspent
    assert "⚠" not in win.results.text()
    win.close()


def test_results_reports_the_model_limit_when_there_is_no_low_energy_law(
        qapp, be10, isobutane_low_p, monkeypatch):
    import core.physics as physics
    from ui.main_window import MainWindow
    monkeypatch.setattr(physics, "_matching_point", lambda mat, part: (0.0, 0.0))
    win = MainWindow()
    win.on_sim_done(run_job(SimJob(isobutane_low_p, [be10], SimSettings())))   # ~14 % unspent again
    assert "⚠" in win.results.text() and "not valid" in win.results.text()
    win.close()


# ---- windows panel ----

def test_windows_panel_rows_order_and_enable(qapp):
    from ui.windows_panel import WindowsPanel
    panel = WindowsPanel()
    panel.add_window("Mylar (C10H8O4)", 0.5, "um")
    panel.add_window("Carbon (amorphous)", 20.0, "ug/cm2")
    assert panel.windows() == []                          # box unchecked: no windows applied
    panel.setChecked(True)
    assert [w.material.name for w in panel.windows()] == ["Mylar (C10H8O4)", "Carbon (amorphous)"]

    panel.table.selectRow(1)
    panel._move_selected(-1)                              # carbon foil first now
    assert [s[0] for s in panel.state()] == ["Carbon (amorphous)", "Mylar (C10H8O4)"]
    assert panel.state()[0] == ("Carbon (amorphous)", 20.0, "ug/cm2")

    panel.table.selectRow(0)
    panel._remove_selected()
    assert panel.state() == [("Mylar (C10H8O4)", 0.5, "um")]


def test_main_window_runs_the_beam_through_windows(qapp, isobutane_low_p, be10):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.p1_panel.input_E0.setValue(1.6)
    win.windows_panel.add_window("Mylar (C10H8O4)", 1.0, "um")
    win.windows_panel.setChecked(True)
    job = win._build_job()
    assert len(job.windows) == 1

    win.on_sim_done(run_job(SimJob(isobutane_low_p, [be10], SimSettings(), windows=job.windows)))
    text = win.results.text()
    assert "Beam: 1.6 MeV" in text and "1. Mylar (C10H8O4), 1 µm" in text and "⚠" not in text
    assert win._results[0].E[0] < 1.6

    thick = win.windows_panel.windows()
    thick[0].thickness = 10.0                             # 10 um of Mylar stops 1.6 MeV Be-10
    win.on_sim_done(run_job(SimJob(isobutane_low_p, [be10], SimSettings(), windows=thick)))
    assert "Stops inside window 1" in win.results.text()
    win.close()


def test_windows_are_saved_with_the_session(qapp):
    from PySide6.QtCore import QSettings
    from ui.main_window import MainWindow
    org, app = "BraggSimTest", "WindowsRoundTrip"
    try:
        win = MainWindow()
        win._SETTINGS_ORG, win._SETTINGS_APP = org, app
        win.windows_panel.add_window("Kapton (C22H10N2O5)", 0.3, "um")
        win.windows_panel.add_window("Carbon (amorphous)", 15.0, "ug/cm2")
        win.windows_panel.setChecked(True)
        win._save_settings()
        win.close()

        fresh = MainWindow()
        fresh._SETTINGS_ORG, fresh._SETTINGS_APP = org, app
        fresh._load_settings()
        assert fresh.windows_panel.isChecked()
        assert fresh.windows_panel.state() == [("Kapton (C22H10N2O5)", 0.3, "um"),
                                               ("Carbon (amorphous)", 15.0, "ug/cm2")]
        fresh.close()
    finally:
        QSettings(org, app).clear()


# ---- layout, anodes, pressure finder ----

def test_window_material_names_have_room(qapp):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    win.tabs.setCurrentIndex(1)
    win.windows_panel.add_window("Kapton (C22H10N2O5)", 0.3, "um")
    qapp.processEvents()
    combo = win.windows_panel.table.cellWidget(0, 0)
    needed = combo.fontMetrics().horizontalAdvance("Kapton (C22H10N2O5)")
    assert combo.width() >= needed + 20, (combo.width(), needed)
    win.close()


def test_anode_columns_follow_the_detector_tab(qapp, isobutane_low_p, be10):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.on_sim_done(run_job(SimJob(isobutane_low_p, [be10], SimSettings())))
    t = win.results.table
    assert not t.isColumnHidden(4) and t.item(0, 4).text() != "—"
    win.detector_panel.box.setChecked(False)
    assert t.isColumnHidden(4) and t.isColumnHidden(5)
    win.close()


def test_pressure_finder_puts_the_crossing_on_the_boundary(qapp):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.material_panel.combo.setCurrentText("Isobutane (C4H10)")
    win.material_panel.set_density(1.08e-5)
    win.p1_panel.combo.setCurrentText("Beryllium-10")
    win.p1_panel.input_E0.setValue(1.6)
    win.p2_panel.combo.setCurrentText("Boron-10")
    win.p2_panel.input_E0.setValue(1.6)
    win.place_crossing_on_boundary()
    rho = win.material_panel.density.value()
    assert rho != pytest.approx(1.08e-5, rel=1e-3)

    job = win._build_job()
    win.on_sim_done(run_job(job))
    assert "on the anode boundary" in win.results.crossing_label.text()
    win.close()


def test_detector_settings_are_saved_with_the_session(qapp):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.detector_panel.gap.setValue(7.5)
    win.detector_panel.anode1.setValue(100.0)
    win.detector_panel.box.setChecked(False)
    win.tabs.setCurrentIndex(2)
    win.close()                                        # closeEvent saves (test settings name)

    fresh = MainWindow()
    fresh._load_settings()
    assert fresh.detector_panel.gap.value() == 7.5
    assert fresh.detector_panel.anode1.value() == 100.0
    assert not fresh.detector_panel.box.isChecked()
    assert fresh.tabs.currentIndex() == 2
    fresh.close()


def test_density_box_keeps_full_precision(qapp):
    from ui.widgets import ScientificSpinBox
    box = ScientificSpinBox()
    box.setRange(1e-12, 100.0)
    box.setSuffix(" g/cm³")
    box.setValue(1.4524337e-05)
    assert box.value() == pytest.approx(1.4524337e-05, rel=1e-9)
    assert box.text() == "1.452434e-05 g/cm³"
    assert box.valueFromText("1,08e-5 g/cm³") == pytest.approx(1.08e-5)
    box.stepBy(1)
    assert box.value() == pytest.approx(1.4524337e-05 * 1.01)


# ---- straggling, dE-E view, no status messages ----

def _be_b_window(qapp):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.material_panel.combo.setCurrentText("Isobutane (C4H10)")
    win.material_panel.set_density(1.08e-5)
    win.p1_panel.combo.setCurrentText("Beryllium-10")
    win.p1_panel.input_E0.setValue(1.6)
    win.p2_panel.combo.setCurrentText("Boron-10")
    win.p2_panel.input_E0.setValue(1.6)
    win.run()
    return win


def test_results_show_widths_and_separation(qapp):
    win = _be_b_window(qapp)
    t = win.results.table
    assert "±" in t.item(0, 2).text() and "±" in t.item(0, 4).text() and "±" in t.item(0, 5).text()
    assert "separation" in win.results.crossing_label.text()
    win.beam_panel.straggling.setChecked(False)
    win.run()
    assert "±" not in t.item(0, 2).text() and "separation" not in win.results.crossing_label.text()
    win.close()


def test_de_e_view_draws_ellipses(qapp):
    from core.units import PlotMode
    win = _be_b_window(qapp)
    win.plot_mode_combo.setCurrentIndex(win.plot_mode_combo.findData(PlotMode.DE_E))
    import pyqtgraph as pg
    curves = [i for i in win.plot.listDataItems()
              if isinstance(i, pg.PlotDataItem) and i.xData is not None and len(i.xData) > 100]
    assert len(curves) == 4                             # 1 and 2 sigma for both particles
    assert not win.xaxis_combo.isEnabled() and not win.unit_combo.isEnabled()
    win.detector_panel.box.setChecked(False)
    assert "Switch on the anodes" in win.results.crossing_label.text()
    win.close()


def test_bad_input_is_reported_in_the_results_not_a_status_bar(qapp):
    win = _be_b_window(qapp)
    win.p1_panel.input_E0.setValue(0.0)
    win.p2_panel.input_E0.setValue(0.0)
    win.run()
    assert "positive E₀" in win.results.crossing_label.text() and not win._results
    from PySide6.QtWidgets import QStatusBar
    assert win.findChild(QStatusBar) is None            # no status bar at all
    win.close()


def test_beam_settings_are_saved_with_the_session(qapp):
    from ui.main_window import MainWindow
    win = MainWindow()
    win.beam_panel.sigma.setValue(12.5)
    win.beam_panel.straggling.setChecked(False)
    win.close()
    fresh = MainWindow()
    fresh._load_settings()
    assert fresh.beam_panel.sigma.value() == 12.5 and not fresh.beam_panel.straggling.isChecked()
    fresh.close()
