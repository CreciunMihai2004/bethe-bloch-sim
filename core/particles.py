"""
Projectile definitions and database
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import U_TO_MEV


@dataclass
class Particle:
    name: str
    z: int          # charge number
    M_u: float      # rest mass in atomic mass units (aum)
    E0: Optional[float] = None  # initial kinetic energy (MeV)
    color: str = field(default_factory=lambda: f"#{random.randint(0, 0xFFFFFF):06x}")

    @property
    def M(self) -> float:
        """Rest mass energy in MeV"""
        return self.M_u * U_TO_MEV

_DATA_DIR = Path(__file__).parent.parent / "data"
_PARTICLES_CSV = _DATA_DIR / "particles.csv"

def _load_particle_db(path: Path = _PARTICLES_CSV) -> dict[str, Particle]:
    """
    Parse path and return a {name: Particle} dict
    (Lines whose first non-whitespace character is ``#`` are treated as
    comments and skipped)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Particle database not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    db: dict[str, Particle] = {}
    errors: list[str] = []

    with open(path, newline="", encoding="utf-8") as fh:
        data_lines = [
            line for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]

    reader = csv.DictReader(data_lines)

    required = {"name", "z", "M_u"}
    if reader.fieldnames and not required.issubset(reader.fieldnames):
        missing = required - set(reader.fieldnames)
        raise ValueError(
            f"{path.name}: missing required columns: {missing}\n"
            f"Found columns: {reader.fieldnames}"
        )

    for row_num, row in enumerate(reader, start=2):
        try:
            name = row["name"].strip()
            if not name:
                continue

            e0_str = row.get("E0", "").strip()
            e0: Optional[float] = float(e0_str) if e0_str else None

            color = row.get("color", "").strip() or "cyan"

            particle = Particle(
                name  = name,
                z     = int(row["z"]),
                M_u   = float(row["M_u"]),
                E0    = e0,
            )
            if name in db:
                errors.append(f"row {row_num}: duplicate name '{name}' — skipped")
                continue
            db[name] = particle
        except (ValueError, KeyError) as exc:
            errors.append(f"row {row_num}: {exc} — skipped")

    if errors:
        import warnings
        warnings.warn(
            f"{path.name} — {len(errors)} row(s) skipped:\n" +
            "\n".join(f"  • {e}" for e in errors),
            stacklevel=2,
        )

    if not db:
        raise ValueError(f"{path.name} contains no valid particle entries.")

    return db


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