from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, TypeVar

T = TypeVar("T")

def get_data_dir() -> Path:
    """Folder holding the editable CSV files (next to the .exe when frozen)"""
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parent.parent

    return base_dir / "data"

def load_csv_db(path: Path,
                required: Iterable[str],
                build: Callable[[dict], T],
                kind: str) -> Dict[str, T]:
    """
    Read a CSV file into a {name: object} dict

    * blank lines and lines starting with '#' are ignored
    * `required` columns must be present in the header
    * build(row) turns one row (a dict of strings) into an object that has
      a `.name`; rows that raise ValueError / KeyError / TypeError, and rows
      with a repeated name, are skipped with a warning that cites the line
      number in the file
    * `kind` ("Material", "Particle") only words the error messages
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} database not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    # keep each line's number in the file so warnings can point at it
    with open(path, newline="", encoding="utf-8") as fh:
        kept = [(no, line) for no, line in enumerate(fh, start=1)
                if line.strip() and not line.strip().startswith("#")]

    reader = csv.DictReader(line for _, line in kept)

    required = set(required)
    if reader.fieldnames and not required.issubset(reader.fieldnames):
        missing = required - set(reader.fieldnames)
        raise ValueError(
            f"{path.name}: missing required columns: {missing}\n"
            f"Found columns: {reader.fieldnames}"
        )

    db: Dict[str, T] = {}
    errors: list[str] = []

    for i, row in enumerate(reader, start=1):       # kept[0] is the header
        line_no = kept[i][0]
        try:
            if not row["name"].strip():
                continue
            obj = build(row)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"line {line_no}: {exc} — skipped")
            continue

        if obj.name in db:
            errors.append(f"line {line_no}: duplicate name '{obj.name}' — skipped")
            continue
        db[obj.name] = obj

    if errors:
        warnings.warn(
            f"{path.name} — {len(errors)} row(s) skipped:\n" +
            "\n".join(f"  • {e}" for e in errors),
            stacklevel=3,
        )

    if not db:
        raise ValueError(f"{path.name} contains no valid {kind.lower()} entries.")

    return db