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


class SimWorker(QObject):
    finished = Signal(object)   # emits List[TrackResult]
    progress = Signal(int)      # 0..100
    error = Signal(str)

    def __init__(self, job: SimJob):
        super().__init__()
        self._job = job

    def run(self):
        try:
            results: List[TrackResult] = []
            n = max(1, len(self._job.particles))

            for i, part in enumerate(self._job.particles):
                # map each particle's 0..1 progress into its slice of 0..100
                def cb(frac, i=i):
                    overall = (i + frac) / n
                    self.progress.emit(int(overall * 100))

                r = simulate(self._job.material, part, self._job.settings, cb)
                results.append(r)

            self.progress.emit(100)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))