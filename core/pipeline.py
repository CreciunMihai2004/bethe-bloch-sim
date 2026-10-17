"""
One simulation run: beam -> windows -> detector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .beamline import Window, simulate_through_windows
from .materials import Material
from .particles import Particle
from .simulation import SimSettings, TrackResult


@dataclass
class SimJob:
    material: Material
    particles: List[Particle]
    settings: SimSettings = field(default_factory=SimSettings)
    windows: List[Window] = field(default_factory=list)     # in beam order
    beam_sigma: float = 0.0                                  # MeV, 1-sigma energy spread of the beam
    straggling: bool = True


def run_track(job: SimJob, part: Particle, material: Material | None = None) -> TrackResult:
    """One particle through the job's windows into its detector (or another material)"""
    return simulate_through_windows(material or job.material, part, job.windows, job.settings,
                                    sigma_beam=job.beam_sigma, straggling=job.straggling)


def run_job(job: SimJob) -> List[TrackResult]:
    return [run_track(job, p) for p in job.particles]
