"""
Projectile definitions and database
"""

from __future__ import annotations

import colorsys
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .constants import U_TO_MEV
from .data_io import get_data_dir, load_csv_db

_DATA_DIR = get_data_dir()
_PARTICLES_CSV = _DATA_DIR / "particles.csv"

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _color_sequence() -> Iterator[str]:
    """Endless supply of well-separated plot colours (golden-ratio hue steps)"""
    for i in itertools.count():
        rgb = colorsys.hsv_to_rgb((_GOLDEN_RATIO_CONJUGATE * i) % 1.0, 1.0, 1.0)
        yield "#" + "".join(f"{int(c * 255):02x}" for c in rgb)


_colors = _color_sequence()


def _next_color() -> str:
    return next(_colors)


@dataclass
class Particle:
    name: str
    z: int          # charge number
    M_u: float      # rest mass in atomic mass units (amu)
    E0: Optional[float] = None  # initial kinetic energy (MeV)
    color: str = field(default_factory=_next_color)

    @property
    def M(self) -> float:
        """Rest mass energy in MeV"""
        return self.M_u * U_TO_MEV


# ---- CSV loader ----

def _build_particle(row: dict) -> Particle:
    e0 = (row.get("E0") or "").strip()

    # Use the CSV colour when given; otherwise the dataclass picks its own
    color = (row.get("color") or "").strip()
    extra = {"color": color} if color else {}

    return Particle(
        name = row["name"].strip(),
        z    = int(row["z"]),
        M_u  = float(row["M_u"]),
        E0   = float(e0) if e0 else None,
        **extra,
    )


def _load_particle_db(path: Path = _PARTICLES_CSV) -> dict[str, Particle]:
    """
    Parse path and return a {name: Particle} dict
    """
    return load_csv_db(path, {"name", "z", "M_u"}, _build_particle, "Particle")


PARTICLE_DB: dict[str, Particle] = _load_particle_db()


def get_particle(name: str) -> Particle:
    try:
        return PARTICLE_DB[name]
    except KeyError:
        raise KeyError(
            f"Particle '{name}' not found.\n"
            f"Available: {list(PARTICLE_DB)}"
        ) from None


def reload() -> None:
    global PARTICLE_DB
    PARTICLE_DB = _load_particle_db()
