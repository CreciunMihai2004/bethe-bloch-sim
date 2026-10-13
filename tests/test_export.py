"""CSV / Excel export"""

import numpy as np
import pandas as pd

from core.simulation import SimSettings, export_csv, export_xlsx, simulate
from core.units import StoppingUnit


def _two_tracks(water, proton):
    from core.particles import Particle
    proton.E0 = 3.0
    alpha = Particle("Helium-4", z=2, M_u=4.0026, E0=12.0)
    return [simulate(water, proton, SimSettings(x_stop=1e4)),
            simulate(water, alpha, SimSettings(x_stop=1e4))]


def test_csv_has_one_x_column_and_both_particles(tmp_path, water, proton):
    res = _two_tracks(water, proton)
    path = tmp_path / "out.csv"
    export_csv(res, str(path), StoppingUnit.LINEAR_MM)
    df = pd.read_csv(path)
    assert df.columns[0] == "x_mm"
    assert sum(c.startswith("E_") for c in df.columns) == 2
    assert df["x_mm"].is_monotonic_increasing


def test_xlsx_has_data_and_metadata_sheets(tmp_path, water, proton):
    res = _two_tracks(water, proton)
    path = tmp_path / "out.xlsx"
    export_xlsx(res, str(path), StoppingUnit.MASS, mass_thickness=True)
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {"Bragg curves", "Metadata"}
    assert len(sheets["Metadata"]) == 2


def test_xlsx_metadata_records_where_and_why_the_track_ended(tmp_path, water, proton):
    res = _two_tracks(water, proton)
    path = tmp_path / "out.xlsx"
    export_xlsx(res, str(path), StoppingUnit.MASS)
    meta = pd.read_excel(path, sheet_name="Metadata")
    assert {"Entry energy (MeV)", "Final energy (MeV)", "Track end"} <= set(meta.columns)
    assert (meta["Final energy (MeV)"] < meta["Entry energy (MeV)"]).all()
    assert set(meta["Track end"]) <= {"particle stopped", "electronic-stopping model limit",
                                      "depth limit (range NOT reached)", "step limit (range NOT reached)"}
