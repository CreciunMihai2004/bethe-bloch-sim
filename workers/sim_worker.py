"""
Background simulation worker
"""

from dataclasses import dataclass
from typing import List

from PySide6.QtCore import QObject, Signal

from core.materials import Material
from core.particles import Particle
from core.simulation import SimSettings, TrackResult, simulate


@dataclass
class SimJob:
    material: Material
    particles: List[Particle]
    settings: SimSettings


class _Cancelled(Exception):
    """Raised inside the progress callback to abort a running simulation"""


class SimWorker(QObject):
    finished = Signal(object)   # emits List[TrackResult]
    progress = Signal(int)      # 0..100, emitted only when the whole-number value changes
    error = Signal(str)

    def __init__(self, job: SimJob):
        super().__init__()
        self._job = job
        self._cancelled = False
        self._last_percent = -1

    def cancel(self):
        """Ask the running simulation to stop at its next progress report"""
        self._cancelled = True

    def _report(self, percent: int):
        if percent != self._last_percent:
            self._last_percent = percent
            self.progress.emit(percent)

    def run(self):
        try:
            results: List[TrackResult] = []
            n = max(1, len(self._job.particles))

            for i, part in enumerate(self._job.particles):
                # map each particle's 0..1 progress into its slice of 0..100
                def cb(frac, i=i):
                    if self._cancelled:
                        raise _Cancelled()
                    self._report(int((i + frac) / n * 100))

                results.append(simulate(self._job.material, part, self._job.settings, cb))

            self._report(100)
            self.finished.emit(results)
        except _Cancelled:
            return      # the window is closing; nothing to report
        except Exception as e:
            self.error.emit(str(e))
