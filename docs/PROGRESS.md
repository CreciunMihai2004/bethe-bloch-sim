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
| 10 | Energy straggling, first part: `core/straggling.py` (Bohr rate with the effective charge, Lindhard-Scharff low-velocity factor, per element), carried from the beam (energy spread input) through every window into the gas: var E = S² J; range straggling, ΔE / E_res widths and their correlation, ΔE-E separation; ΔE vs E_res plot with 1σ / 2σ ellipses, ±1σ band in the energy plot, widths in the results and the Excel metadata. Clean-up: no status bar (problems show in the results line), runs are synchronous (the worker thread, progress and cancel code are gone), export in `core/export.py`, one `get_base_dir()` | σ_E, correlation and range spread within 0.5 % of a brute-force Monte Carlo (20 000 ions) |
| 11 | Mean excitation energy from the formula: ln I = Σ nᵢZᵢ ln Iᵢ / Σ nᵢZᵢ with the ICRU 37 bound-atom values (gas: H 19.2, C 70 eV; condensed: H 19.2, C 81, N 82, O 106, F 112, Cl 180 eV, others 1.13 × elemental); Z and A summed over the formula too, so `materials.csv` / `windows.csv` need only name, density and formula (a given Z, A or I still wins) | methane 41.72 (41.7), Mylar 78.68 (78.7), Kapton 79.58 (79.6), polypropylene 56.52 (56.5), NaI 452.0 (452.0) eV; isobutane 47.85 eV vs ICRU 49's 48.3 eV: Be-10 range 385.07 → 382.63 mm (−0.63 %) |

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
| 2026-10-17 | Be-10 1.6 MeV ± 10 keV, energy spread at 50 / 110 / 200 / 300 mm | 10.62 / 10.47 / 8.43 / 5.26 keV | Monte Carlo, 20 000 ions: 10.57 / 10.44 / 8.43 / 5.25 keV | propagation formula checked |
| 2026-10-17 | range straggling from electronic straggling vs SRIM longitudinal straggling | Be-10 1.6 MeV: 1.62 mm; B-10 1.4 MeV: 1.18 mm | SRIM: 14.51 / 9.99 mm | ratio ~0.11 at all energies: SRIM's projected-range straggling is dominated by nuclear scattering (path zig-zag, lateral 18 mm), not in the model yet |

## Next (in order)

1. **Straggling, second part**: charge-exchange straggling (charge-state fluctuations of heavy ions in gas; Yang et al. 1991 empirical form), window thickness non-uniformity, detector resolution (ion-pair statistics, noise). Then the ΔE-E widths become realistic and the separation meaningful.
2. **Monte Carlo** with nuclear scattering (angles, path length): needed for range / lateral straggling against SRIM, and for the ΔE-E clouds themselves.
3. **Heavy-ion electronic stopping at low velocity** (B-10 gap vs SRIM): options (a) report as a limitation, (b) effective-charge scaling of proton stopping in C and H (ICRU-49 / PSTAR), (c) correction tables. To decide.
4. **Real beamline data**: window materials and thicknesses, entrance gap, beam energy spread.
5. Pressure finder that maximises the ΔE-E separation (once 1 is in), accelerator stages.

## Open questions

* Window materials and thicknesses?
* Measured data (spectrum clouds)