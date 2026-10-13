# BraggSim: progress log

Thesis working title: *Simulation and optimization of detector gas pressure for isobaric separation
of ¹⁰Be and ¹⁰B in AMS at a 1 MV Tandetron (IFIN-HH)*.

## Done

| # | Item | Evidence |
|---|------|----------|
| 1 | pytest suite (physics, integrator, intersections, CSV, export, GUI smoke) | `python -m pytest` |
| 2 | Cleanups: numpy intersection finder (identical results, ~8× faster), in-place DB reload, CODATA 2018 constants | tests |
| 3 | ZBL nuclear stopping restored to the published form (`1.1383` factor, `Z^0.23` screening, ε > 30 branch) | s_n peaks at ε = 0.327; matches the analytic conversion |
| 4 | Model limit is explicit: the track stops where Bethe-Bloch is invalid, the UI and Excel metadata say so, and the 1.5 %-of-peak trim heuristic is gone | range unchanged (372.7 mm Be-10) |

## Validation log

| Date | Case | App | Reference | Note |
|------|------|-----|-----------|------|
| 2026-10-13 | p in water, 100 MeV | 7.293 | 7.289 (PSTAR) | +0.06 % |
| 2026-10-13 | p in water, 10 MeV | 45.97 | 45.67 (PSTAR) | +0.65 % |
| 2026-10-13 | p in water, 1 MeV | 268.1 | 260.8 (PSTAR) | +2.8 %, no shell correction |
| 2026-10-13 | Be-10 1.6 MeV, n-butane 1.08e-5 g/cm³ | 372.7 mm, **226 keV untracked** | 368.3 mm (SRIM, projected) | agreement includes an untracked tail |
| 2026-10-13 | B-10 1.4 MeV, same gas | 273.4 mm, 226 keV untracked | 242.55 mm (SRIM) | +12.7 % |

## Next (in order)

1. **Low-energy electronic stopping** (removes the 226 keV leftover). Needs Bragg additivity (C and H separately), a Lindhard-Scharff regime and a heavy-ion effective charge. Needs the SRIM stopping tables to calibrate and to check range vs energy, not just the final range.
2. Material composition (`formula` column) so nuclear and low-energy stopping are summed per element.
3. Beamline as a chain of elements (stripper gas, windows, detector) with energy in at the source.
4. Straggling, then Monte Carlo clouds.
5. Pressure optimization for the ¹⁰Be/¹⁰B separation.

## Open questions

* Window materials and thicknesses; is there a degrader foil?
* Is measured data available (even a few spectra)?
* Thesis length and what the supervisor wants (software or analysis).
