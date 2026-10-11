"""
Read-only diagnostic for the stopping-power model.

    python tools/diagnose_stopping.py --material "Isobutane (C4H10)" \
        --particle Beryllium-10 --E0 1.6 --srim "Beryllium in Butane.txt"

--srim takes the raw .txt that SRIM writes (any stopping units, decimal comma or
point).  Without it the script still compares the old and the new model.

Prints
  1. how the material was split into atoms (ZBL / Lindhard need per-atom values)
  2. electronic and nuclear stopping versus energy for the legacy and blend
     models, the Bethe / Lindhard-Scharff curves the blend is made from and,
     with --srim, the SRIM values and the model/SRIM ratios
  3. a full track with each model (and with the SRIM table) : range, energy
     left at the end, peak
  4. with --srim, the range SRIM itself gives (projected, and CSDA from the table)

--density defaults to the density of the SRIM run (so the ranges are directly
comparable) and otherwise to the one in materials.csv.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import physics as ph                      # noqa: E402
from core.materials import get_material             # noqa: E402
from core.particles import get_particle             # noqa: E402
from core.simulation import SimSettings, simulate   # noqa: E402
from core.srim_table import parse_srim_table        # noqa: E402
from core.units import StoppingUnit                 # noqa: E402


def _track_line(label, r, E0):
    return (f"  {label:7s}: range = {r.range_mm:9.3f} mm, "
            f"energy left = {r.E[-1]:.4f} MeV ({100 * r.E[-1] / E0:.2f} % of E0), "
            f"peak = {r.peak_dEdx(StoppingUnit.LINEAR_MM):.4g} MeV/mm, points = {len(r.E)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", required=True)
    ap.add_argument("--particle", required=True)
    ap.add_argument("--E0", type=float, required=True, help="initial energy (MeV)")
    ap.add_argument("--density", type=float, default=None, help="override density (g/cm3)")
    ap.add_argument("--srim", default=None, help="SRIM output .txt for this ion in this material")
    args = ap.parse_args()

    mat = deepcopy(get_material(args.material))
    part = deepcopy(get_particle(args.particle))
    part.E0 = args.E0
    table = parse_srim_table(args.srim) if args.srim else None

    if args.density:
        mat.rho = args.density
    elif table and table.density > 0:
        mat.rho = table.density
        print(f"(using the SRIM run's density, {mat.rho:g} g/cm3; override with --density)")

    # ---- 1. composition ----
    print(f"Material  : {mat.name}   rho = {mat.rho:g} g/cm3   I = {mat.I_eV} eV")
    print(f"CSV values: Z = {mat.Z:g}, A = {mat.A:g} g/mol   (Z/A = {mat.Z / mat.A:.4f})")
    print(f"Composition used for ZBL / Lindhard-Scharff ({mat.composition_source}):")
    for Z2, M2, n in mat.components:
        print(f"    {n:g} x atom with Z = {Z2:g}, M = {M2:.4g}")
    if mat.composition_source == "single-atom" and mat.A > 20 and mat.Z > 10:
        print("    WARNING: Z and A look like a molecule but no formula was found; "
              "add a formula in the name, e.g. 'Name (C4H10)', or a 'formula' column")
    Ex = ph.crossover_energy(mat, part)
    print(f"Crossover energy where Bethe = Lindhard-Scharff: {Ex:.4g} MeV ({Ex / part.M_u * 1e3:.3g} keV/u)")
    if table:
        print(f"SRIM table: {table.ion} [{table.Z1}], {table.mass_u:g} amu, "
              f"{table.E_min:g} - {table.E_max:g} MeV, {len(table.E)} rows ({table.source})")
        if abs(table.Z1 - part.z) > 0:
            print(f"    WARNING: table is for Z = {table.Z1} but the particle has z = {part.z}")
        if args.E0 > table.E_max:
            print(f"    NOTE: E0 = {args.E0:g} MeV is above the table; above {table.E_max:g} MeV the "
                  "blend model (scaled to the table) is used. Export the SRIM table up to E0.")
    print()

    # ---- 2. stopping power versus energy ----
    if table:
        energies = np.unique(np.concatenate([table.E[::3], [table.E_min, table.E_max]]))
        energies = energies[energies <= args.E0 * 1.0001]
    else:
        energies = np.geomspace(0.002, args.E0 * 1.5, 26)

    hdr = (f"{'E (MeV)':>9} {'E/u keV':>8} | {'Se old':>8} {'Sn old':>7} | "
           f"{'Se new':>8} {'Sn new':>7} | {'Bethe':>8} {'LS':>8}")
    if table:
        hdr += f" | {'Se SRIM':>8} {'Sn SRIM':>8} {'Se new/S':>8} {'Sn new/S':>8} {'Se old/S':>8}"
    print("Stopping power, MeV cm2/g   (old = legacy model, new = blend model)")
    print(hdr)
    print("-" * len(hdr))
    for E in energies:
        eo, no = ph.dEdx_components(mat, part, E, "legacy")
        en, nn = ph.dEdx_components(mat, part, E, "blend")
        line = (f"{E:9.4f} {E / part.M_u * 1e3:8.1f} | {eo:8.1f} {no:7.1f} | "
                f"{en:8.1f} {nn:7.1f} | {ph.bethe_stopping(mat, part, E):8.1f} "
                f"{ph.lindhard_scharff_stopping(E, part, mat):8.1f}")
        if table and table.E_min <= E <= table.E_max:
            se, sn = table.Se_at(E), table.Sn_at(E)
            line += f" | {se:8.1f} {sn:8.1f} {en / se:8.3f} {nn / sn:8.3f} {eo / se:8.3f}"
        print(line)

    if table:
        m = (table.E >= table.E_min) & (table.E <= min(args.E0, table.E_max))
        ls = np.array([ph.lindhard_scharff_stopping(e, part, mat) for e in table.E[m]])
        low = table.E[m] <= 0.1
        if low.any():
            r = table.Se[m][low] / ls[low]
            print(f"\nSRIM Se / Lindhard-Scharff below 100 keV: {r.min():.2f} - {r.max():.2f} "
                  f"(median {np.median(r):.2f}); the blend model uses 1.00")

    leg_zero = [E for E in np.geomspace(1e-3, args.E0, 400)
                if ph.dEdx_components(mat, part, E, "legacy")[0] == 0.0]
    if leg_zero:
        print(f"Legacy electronic term is exactly zero below {max(leg_zero):.4g} MeV")

    # ---- 3. tracks ----
    print("\nTracks (max_dx = 0.01 mm, frac_loss = 1 %, E_cutoff = 1 keV)")
    old_model = ph.get_default_model()
    try:
        runs = [("legacy", True), ("blend", False)]
        if table:
            ph.clear_srim_tables()
            ph.register_srim_table(part, mat, table)
            runs.append(("srim", False))
        for model, trim in runs:
            ph.set_default_model(model)
            r = simulate(mat, part, SimSettings(x_stop=1e5, trim_tail=trim))
            print(_track_line(model, r, args.E0))
    finally:
        ph.set_default_model(old_model)
        ph.clear_srim_tables()

    # ---- 4. what SRIM itself says ----
    if table and args.E0 >= table.E_min:
        E0c = min(args.E0, table.E_max)
        i = np.searchsorted(table.E, E0c)
        i = min(i, len(table.E) - 1)
        print(f"\nSRIM projected range at {table.E[i]:g} MeV : {table.Rp_mm[i]:.3f} mm")
        sel = table.E <= table.E[i]
        sub = type(table)(table.E[sel], table.Se[sel], table.Sn[sel], table.Rp_mm[sel],
                          table.density)
        print(f"CSDA from the table (10 keV -> E0 by integration + SRIM range below) : "
              f"{sub.csda_mm(mat.rho) + table.Rp_mm[0] * table.density / mat.rho:.3f} mm")
        print("(the projected range is shorter than the CSDA path; the two differ by ~1-2 % here)")


if __name__ == "__main__":
    main()
