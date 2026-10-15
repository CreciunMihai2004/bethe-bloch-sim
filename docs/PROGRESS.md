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

## Validation log

| Date | Case | App | Reference | Note |
|------|------|-----|-----------|------|
| 2026-10-13 | p in water, 100 MeV | 7.293 | 7.289 (PSTAR) | +0.06 % |
| 2026-10-13 | p in water, 10 MeV | 45.97 | 45.67 (PSTAR) | +0.65 % |
| 2026-10-13 | p in water, 1 MeV | 268.1 | 260.8 (PSTAR) | +2.8 %, no shell correction |
| 2026-10-13 | Be-10 1.6 MeV, n-butane 1.08e-5 g/cm³ | 372.7 mm, **226 keV untracked** | 368.3 mm (SRIM)
| 2026-10-13 | B-10 1.4 MeV, same gas | 273.4 mm, 226 keV untracked | 242.55 mm (SRIM) | +12.7 % |

## Next (in order)

1. **Low-energy electronic stopping** (removes the 226 keV leftover). Needs Bragg additivity (C and H separately), a Lindhard-Scharff regime and a heavy-ion effective charge. Needs the SRIM stopping tables to calibrate and to check range vs energy.
2. Material composition (`formula` column) so nuclear and low-energy stopping are summed per element.
3. Beamline as a chain of elements (stripper gas, windows, detector) with energy in at the source.
4. Straggling, then Monte Carlo clouds.

## Open questions

* Window materials and thicknesses?
* Measured data (spectrum clouds)