"""CSV databases"""

import warnings

import pytest

from core.data_io import load_csv_db
from core.materials import MATERIAL_DB, _build_material, get_material
from core.particles import PARTICLE_DB, get_particle


def test_shipped_databases_load_without_warnings():
    assert "Isobutane (C4H10)" in MATERIAL_DB
    for name in ("Beryllium-10", "Boron-10", "Carbon-14"):
        assert name in PARTICLE_DB


def test_unknown_names_raise_helpful_errors():
    with pytest.raises(KeyError, match="not found"):
        get_material("Unobtainium")
    with pytest.raises(KeyError, match="not found"):
        get_particle("Unobtainium")


def test_bad_rows_are_skipped_with_a_line_number(tmp_path):
    csv_file = tmp_path / "m.csv"
    csv_file.write_text(
        "# comment\n"
        "name,Z,A,I_eV,rho\n"
        "Good,1,1.0,19.2,0.1\n"
        "Bad,notanumber,1.0,19.2,0.1\n"
        "Good,2,4.0,41.8,0.2\n"          # duplicate name
    )
    with pytest.warns(UserWarning, match="2 row"):
        db = load_csv_db(csv_file, {"name", "Z", "A", "I_eV", "rho"}, _build_material, "Material")
    assert list(db) == ["Good"]


def test_missing_columns_are_reported(tmp_path):
    csv_file = tmp_path / "m.csv"
    csv_file.write_text("name,Z\nX,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv_db(csv_file, {"name", "Z", "A"}, _build_material, "Material")


def test_material_with_short_row_uses_sternheimer_defaults():
    si = get_material("Silicon (Si)")
    assert (si.X0, si.X1) == (1.6, 4.0)


def test_reload_keeps_the_same_dict_object():
    """Modules do `from core.materials import MATERIAL_DB`; reload must update that very dict"""
    import core.materials as m
    import core.particles as p
    mdb, pdb = m.MATERIAL_DB, p.PARTICLE_DB
    m.reload()
    p.reload()
    assert m.MATERIAL_DB is mdb and p.PARTICLE_DB is pdb
    assert "Isobutane (C4H10)" in mdb and "Beryllium-10" in pdb


def test_constants_are_codata_2018():
    from core import constants as c
    assert c.N_A == pytest.approx(6.02214076e23, rel=1e-9)
    assert c.ME_C2 == pytest.approx(0.51099895, rel=1e-8)
