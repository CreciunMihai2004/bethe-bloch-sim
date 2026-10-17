"""The Qt window builds, simulates in its worker, and draws"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from core.simulation import SimSettings
from workers.sim_worker import SimJob, SimWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_worker_runs_a_job(qapp, water, proton):
    proton.E0 = 5.0
    worker = SimWorker(SimJob(water, [proton], SimSettings()))
    got = []
    worker.finished.connect(got.append)
    worker.run()                                   # synchronous: no thread needed in a test
    assert len(got) == 1 and len(got[0]) == 1


def test_main_window_draws_results(qapp, water, proton):
    from ui.main_window import MainWindow
    win = MainWindow()
    proton.E0 = 5.0
    job = SimJob(water, [proton], SimSettings())
    worker = SimWorker(job)
    worker.finished.connect(win.on_sim_done)
    worker.run()
    assert win._results
    assert "range" in win.results_label.text()
    win.close()


def test_edited_particle_is_labelled_custom(qapp):
    from ui.main_window import ParticlePanel
    panel = ParticlePanel("P", "Beryllium-10")
    assert panel.particle().name == "Beryllium-10"
    panel.spin_z.setValue(panel.spin_z.value() + 1)
    assert panel.particle().name == "Beryllium-10 (custom)"


def test_results_label_warns_when_track_is_cut(qapp, water, proton):
    from ui.main_window import MainWindow
    win = MainWindow()
    proton.E0 = 100.0
    worker = SimWorker(SimJob(water, [proton], SimSettings(x_stop=10.0)))
    worker.finished.connect(win.on_sim_done)
    worker.run()
    assert "cut" in win.results_label.text().lower()
    win.close()


def test_results_label_has_no_warning_for_a_track_that_runs_to_rest(qapp, water, proton, be10, isobutane_low_p):
    from ui.main_window import MainWindow
    win = MainWindow()

    proton.E0 = 5.0
    w = SimWorker(SimJob(water, [proton], SimSettings()))
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "⚠" not in win.results_label.text()

    w = SimWorker(SimJob(isobutane_low_p, [be10], SimSettings()))     # used to warn: ~14 % unspent
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "⚠" not in win.results_label.text()
    win.close()


def test_results_label_reports_the_model_limit_when_there_is_no_low_energy_law(
        qapp, be10, isobutane_low_p, monkeypatch):
    import core.physics as physics
    from ui.main_window import MainWindow
    monkeypatch.setattr(physics, "_matching_point", lambda mat, part: (0.0, 0.0))
    win = MainWindow()
    w = SimWorker(SimJob(isobutane_low_p, [be10], SimSettings()))     # ~14 % unspent again
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "⚠" in win.results_label.text() and "not valid" in win.results_label.text()
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
    job = win._build_job(show_warnings=False)
    assert len(job.windows) == 1

    w = SimWorker(SimJob(isobutane_low_p, [be10], SimSettings(), windows=job.windows))
    w.finished.connect(win.on_sim_done)
    w.run()
    text = win.results_label.text()
    assert "windows:" in text and "into the detector" in text and "⚠" not in text
    assert win._results[0].E[0] < 1.6

    thick = win.windows_panel.windows()
    thick[0].thickness = 10.0                             # 10 um of Mylar stops 1.6 MeV Be-10
    w = SimWorker(SimJob(isobutane_low_p, [be10], SimSettings(), windows=thick))
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "stops inside window 1" in win.results_label.text()
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
