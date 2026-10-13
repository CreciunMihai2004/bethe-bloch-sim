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


def test_results_label_reports_the_model_limit_only_when_it_matters(qapp, water, proton, be10, isobutane_low_p):
    from ui.main_window import MainWindow
    win = MainWindow()

    proton.E0 = 5.0                                  # ~0.4 % unspent: not worth a warning
    w = SimWorker(SimJob(water, [proton], SimSettings(x_stop=1e4)))
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "⚠" not in win.results_label.text()

    w = SimWorker(SimJob(isobutane_low_p, [be10], SimSettings()))     # ~14 % unspent
    w.finished.connect(win.on_sim_done)
    w.run()
    assert "⚠" in win.results_label.text() and "not valid" in win.results_label.text()
    win.close()
