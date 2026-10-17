# BraggSim: progress log

Thesis working title: *Numerical Modelling of Heavy-Ion Energy Loss and Bragg Curves in Gas Ionization Detectors for Accelerator Mass Spectrometry*.

## Done

| # | Item | Evidence |
|---|------|----------|
| 1 | pytest suite (physics, integrator, intersections, CSV, export, GUI smoke) | `python -m pytest` |
| 2 | Cleanups: numpy intersection finder (identical results, ~8× faster), in-place DB reload, CODATA 2018 constants | tests |
| 3 | ZBL nuclear stopping restored to the published form (`1.1383` factor, `Z^0.23` screening, ε > 30 branch) | s_n peaks at ε = 0.327; matches the analytic conversion |
| 4 | Model limit is explicit: the track stops where Bethe-Bloch is invalid, the UI and Excel metadata say so, and the 1.5 %-of-peak trim heuristic is gone | range unchanged (372.7 mm Be-10) |
| 5 | Speed: stopping power is vectorized and the integrator walks a log-energy grid instead of a 0.01 mm stepping loop (Be-10 in gas 333 ms -> 0.7 ms, p 100 MeV in water 94 ms -> 0.8 ms; curves have ~800-3000 points instead of up to 37 000); the track ends exactly at the electronic-stopping limit; auto-run debounce 350 -> 80 ms | ranges within 0.12 % of the old integrator (Be-10 372.73 mm, B-10 273.43 mm), `python -m pytest` |
| 6 | Low-energy electronic stopping: Bethe-Bloch above E_m, a velocity-proportional law (S ~ sqrt(E)) below, joined in value and slope (E_m is where Bethe-Bloch's own log-slope falls to 1/2, so no fitted parameter); nuclear and low-energy stopping are summed per element through a new `formula` column in `materials.csv` (Bragg additivity); the track now ends at exactly E = 0 (`stopped`); the 1000 mm depth limit is gone (`SimSettings.x_stop = None` by default, a number still works) | 0 keV unspent (was 226 keV); Be-10 385.1 mm, B-10 278.7 mm; 74 tests |
| 7 | SRIM benchmark: `core/srim.py` reads SRIM-2013 stopping/range tables (decimal commas or points); the institute's tables (Be-10, B-10 in n-butane, 1.08e-5 g/cm³, 10 keV-2 MeV) are in `tests/data` with benchmark tests | nuclear stopping within 1.5 % of SRIM at every energy; path-length bands guarded |
| 8 | Windows, first part of the beamline: `core/beamline.py` passes the beam through an ordered list of windows (µm, nm, µg/cm², mg/cm²) with the same stopping model; the detector track starts at the energy that comes out; a particle can stop in a window. Window materials in `data/windows.csv` (Mylar, Kapton, polypropylene, amorphous carbon; NIST ESTAR I values) | E_out reproduces R(E_in) − t within 0.5 %; thin-window loss = S·t |
| 9 | Anodes and UI reorganisation: `core/detector.py` (anode layout, ΔE / E_res per particle, pressure finder that puts the dE/dx crossing on the anode boundary, ρ = 10·X_cross / x_boundary); tabs Setup / Beamline / Detector, display row above the plot, File menu, compact results table (details in tooltips), anode bands on the plot, anode signals in the Excel metadata; density box with significant digits (the old 8-decimal box kept 3 digits at 1e-5); `get_material` / `get_particle` return copies | crossing within 1 µm of the boundary; ΔE + E_res + gap + escaped = E_in |

## Validation log

| Date | Case | App | Reference | Note |
|------|------|-----|-----------|------|
| 2026-10-13 | p in water, 100 MeV | 7.293 | 7.289 (PSTAR) | +0.06 % |
| 2026-10-13 | p in water, 10 MeV | 45.97 | 45.67 (PSTAR) | +0.65 % |
| 2026-10-13 | p in water, 1 MeV | 268.1 | 260.8 (PSTAR) | +2.8 %, no shell correction |
| 2026-10-13 | Be-10 1.6 MeV, n-butane 1.08e-5 g/cm³ | 372.7 mm, **226 keV untracked** | 368.3 mm (SRIM)
| 2026-10-13 | B-10 1.4 MeV, same gas | 273.4 mm, 226 keV untracked | 242.55 mm (SRIM) | +12.7 % |
| 2026-10-16 | Be-10 1.6 MeV, same gas, low-energy model | 385.1 mm, 0 keV unspent | 368.3 mm (SRIM, projected range) | +4.6 %; the app integrates path length, which is a little longer than projected range |
| 2026-10-16 | B-10 1.4 MeV, same gas, low-energy model | 278.7 mm, 0 keV unspent | 242.55 mm (SRIM, projected range) | +14.9 %: the weak case, see Next |
| 2026-10-16 | low-energy S_e amplitude vs Lindhard-Scharff (no fit) | Be-10 0.86, B-10 0.94 of LS | LS formula, eV/1e15 at/cm2 | both below LS, so the range is not too long because of a too-weak amplitude alone |
| 2026-10-17 | Be-10 and B-10 nuclear stopping vs SRIM table, 10 keV-2 MeV | | SRIM-2013 | within 1.5 % everywhere (ZBL summed over C and H) |
| 2026-10-17 | Be-10 path length 10 keV → 1.6 MeV | 376.8 mm | 367.3 mm from SRIM's own S columns | +2.6 % (like-for-like; SRIM projected/path = 0.986) |
| 2026-10-17 | B-10 path length 10 keV → 1.4 MeV | 272.1 mm | 241.4 mm from SRIM's own S columns | +12.8 % |
| 2026-10-17 | Be-10 S_e app/SRIM | 0.81 (10-20 keV), 0.88-0.95 (50-200 keV), 1.00-1.03 (300 keV-1 MeV), 0.90 (1.6 MeV) | SRIM-2013 | errors partly cancel in the range |
| 2026-10-17 | B-10 S_e app/SRIM | 0.63-0.66 (10-50 keV), 0.75-0.87 (100-200 keV), 0.91-0.95 (300 keV-1.4 MeV) | SRIM-2013 | the cause of the B-10 range gap |

## Next (in order)

1. **Heavy-ion electronic stopping at low velocity.** At the same velocity (50 keV) SRIM has B stopping 1.72x as hard as Be; a smooth Lindhard-Scharff-type law gives 1.14x (Z1 oscillation, empirical). Options: (a) report the gap as a validated limitation; (b) effective-charge scaling of *proton* stopping in C and H (ICRU-49 / PSTAR data, Brandt-Kitagawa), the way SRIM is built; (c) optional per-ion correction tables. To decide.
2. **Real beamline data**: window materials and thicknesses, entrance gap before anode 1, degrader question.
3. **Straggling**: energy straggling in windows and gas (Bohr, charge-state), then widths of ΔE and E_res; SRIM's straggling columns are the benchmark.
4. Monte Carlo ΔE-E clouds (needs 3 and the anodes), then accelerator stages.

## Open questions

* Window materials and thicknesses?
* Measured data (spectrum clouds)