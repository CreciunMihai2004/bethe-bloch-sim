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

## Next (in order)

1. **Calibrate the heavy-ion electronic stopping.** The low-energy law gets its amplitude from Bethe-Bloch with the Barkas effective charge, which is only an approximation at 0.1 MeV/u. B-10 is +14.9 % over SRIM and Be-10 +4.6 %; SRIM's ion-specific fits are what the app does not have. Needs the SRIM range-vs-energy tables (Be-10 and B-10 in n-butane) to see where the two curves part, then a heavy-ion effective-charge / Z1-dependence model validated against them.
2. Beamline as a chain of elements (stripper gas, windows, detector) with energy in at the source.
3. Straggling, then Monte Carlo clouds.

## Open questions

* Window materials and thicknesses?
* Measured data (spectrum clouds)