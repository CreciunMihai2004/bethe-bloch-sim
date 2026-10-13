"""
Simulation and post-processing
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .materials import Material
from .particles import Particle
from .physics import dEdx_components
from .units import StoppingUnit, convert_from_mass

_MAX_STEPS = 500_000       # safety cap on the number of integration steps
_PROGRESS_EVERY = 0.002    # report progress every 0.2 % of the energy lost
_MIN_STEP_FACTOR = 1e-4    # smallest allowed step, as a fraction of max_dx


@dataclass
class SimSettings:
    x_start: float = 0.0        # mm
    x_stop: float = 1000.0      # mm
    E_cutoff: float = 0.001     # MeV
    max_dx: float = 0.01        # mm
    frac_loss: float = 0.01     # max fractional energy loss per step


@dataclass
class TrackResult:
    """
    Bragg curve result: the track from the entry energy down to where the
    stopping-power model stops being valid (or the particle stops)

    x_mass    : mass thickness (g/cm^2)
    E         : kinetic energy (MeV)
    dEdx_mass : mass stopping power (MeV cm^2 / g)
    rho       : material density (g/cm^3)
    stop_reason : why the track ended
        "stopped"     the particle ran out of energy (range reached)
        "model_limit" electronic stopping (Bethe-Bloch) is no longer valid below
                      this energy, so the track stops here; the remaining
                      slowing-down is NOT in range_mm (see unspent_fraction)
        "x_stop"      the depth limit of SimSettings was hit first
        "max_steps"   the step cap was hit first
        With "x_stop" or "max_steps", range_mm is only a lower bound.
    """
    name: str
    color: str
    x_mass: np.ndarray
    E: np.ndarray
    dEdx_mass: np.ndarray
    rho: float
    stop_reason: str = "stopped"

    @property
    def is_complete(self) -> bool:
        """False when the depth or step limit cut the track short"""
        return self.stop_reason in ("stopped", "model_limit")

    @property
    def unspent_fraction(self) -> float:
        """Share of the entry energy still left at the end of the track (0 = fully slowed down)"""
        return float(self.E[-1] / self.E[0]) if self.E[0] > 0 else 0.0

    @property
    def x_mm(self) -> np.ndarray:
        return self.x_mass / self.rho * 10.0

    def dEdx_in(self, unit: StoppingUnit) -> np.ndarray:
        return convert_from_mass(self.dEdx_mass, unit, self.rho)

    def x_in(self, mass_thickness: bool = False) -> np.ndarray:
        return self.x_mass if mass_thickness else self.x_mm

    @property
    def range_mm(self) -> float:
        return float(self.x_mm[-1])

    @property
    def range_mass(self) -> float:
        return float(self.x_mass[-1])

    def peak_dEdx(self, unit: StoppingUnit) -> float:
        return float(np.nanmax(self.dEdx_in(unit)))


# ---- integrator ----

def simulate(mat: Material,
             part: Particle,
             settings: SimSettings,
             progress_cb: Optional[Callable[[float], None]] = None) -> TrackResult:

    if part.E0 is None or part.E0 <= 0:
        raise ValueError(f"{part.name}: initial kinetic energy must be > 0 MeV.")

    mm_to_mass = mat.rho / 10.0
    x_stop_m   = settings.x_stop * mm_to_mass
    max_dx_m   = settings.max_dx * mm_to_mass
    min_dx_m   = max_dx_m * _MIN_STEP_FACTOR

    E0 = float(part.E0)
    x, E = settings.x_start * mm_to_mass, E0
    x_vals, E_vals, dEdx_vals = [x], [E], []

    steps = 0
    last_reported = -1.0
    stop_reason = None

    while x < x_stop_m and E > settings.E_cutoff and steps < _MAX_STEPS:
        steps += 1
        elec1, nuc1 = dEdx_components(mat, part, E)
        s1 = elec1 + nuc1
        if elec1 <= 0:
            # Bethe-Bloch is not valid any more
            stop_reason = "model_limit"
            break

        dx = max(min(max_dx_m, settings.frac_loss * E / s1), min_dx_m)

        # midpoint rule: use the stopping power half-way through the step
        E_mid = E - 0.5 * dx * s1
        elec_m, nuc_m = dEdx_components(mat, part, E_mid) if E_mid > 0 else (0.0, 0.0)
        s_mid = elec_m + nuc_m
        if elec_m <= 0:
            # this step would cross the validity limit: end the track at its start
            dEdx_vals.append(s1)
            stop_reason = "model_limit"
            break

        dEdx_vals.append(s_mid)
        x += dx
        E = max(E - dx * s_mid, 0.0)
        x_vals.append(x)
        E_vals.append(E)

        if progress_cb is not None:
            frac = min(1.0, (E0 - E) / E0)
            if frac - last_reported >= _PROGRESS_EVERY:
                progress_cb(frac)
                last_reported = frac

    if stop_reason is None:
        if x >= x_stop_m and E > settings.E_cutoff:
            stop_reason = "x_stop"
        elif steps >= _MAX_STEPS and E > settings.E_cutoff:
            stop_reason = "max_steps"
        else:
            stop_reason = "stopped"

    if len(dEdx_vals) < len(x_vals):
        dEdx_vals.append(dEdx_vals[-1] if dEdx_vals else float("nan"))

    if progress_cb is not None:
        progress_cb(1.0)

    return TrackResult(
        name=part.name,
        color=part.color,
        x_mass=np.array(x_vals),
        E=np.array(E_vals),
        dEdx_mass=np.array(dEdx_vals),
        rho=mat.rho,
        stop_reason=stop_reason,
    )


# ---- intersection finder ----

def curve_crossings(xa: np.ndarray, ya: np.ndarray,
                    xb: np.ndarray, yb: np.ndarray) -> List[tuple]:
    """
    Points where two piecewise-linear curves cross, as (x, y) tuples
    """
    xa, ya = np.asarray(xa, float), np.asarray(ya, float)
    xb, yb = np.asarray(xb, float), np.asarray(yb, float)
    ka, kb = np.isfinite(ya), np.isfinite(yb) 
    xa, ya, xb, yb = xa[ka], ya[ka], xb[kb], yb[kb]
    if len(xa) < 2 or len(xb) < 2:
        return []

    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    if lo >= hi:
        return []

    grid = np.union1d(xa, xb)
    grid = grid[(grid >= lo) & (grid <= hi)]
    fa = np.interp(grid, xa, ya)
    fb = np.interp(grid, xb, yb)
    diff = fa - fb

    nonneg = diff >= 0
    crossings: List[tuple] = []
    for i in np.where(nonneg[:-1] != nonneg[1:])[0]:
        d0, d1 = diff[i], diff[i + 1]
        t = d0 / (d0 - d1)
        x = grid[i] + t * (grid[i + 1] - grid[i])
        y = fa[i] + t * (fa[i + 1] - fa[i])
        if not crossings or not np.isclose(x, crossings[-1][0], rtol=1e-9, atol=0.0):
            crossings.append((float(x), float(y)))
    return crossings


def find_intersections(a: TrackResult, b: TrackResult,
                       unit: StoppingUnit,
                       mass_thickness: bool = False) -> List[tuple]:
    """Crossings of the dE/dx curves of two tracks, in the displayed units"""
    return curve_crossings(a.x_in(mass_thickness), a.dEdx_in(unit),
                           b.x_in(mass_thickness), b.dEdx_in(unit))


# ---- export ----

def _build_export_df(results: List[TrackResult], unit: StoppingUnit,
                     mass_thickness: bool, unit_in_name: bool = False) -> pd.DataFrame:
    """
    Shared table-building logic for CSV and Excel export
    """
    x_label = "x_g_per_cm2" if mass_thickness else "x_mm"
    suffix = f"_{unit.label}" if unit_in_name else ""

    labels: List[str] = []
    for r in results:
        label, k = r.name, 2
        while label in labels:
            label, k = f"{r.name}#{k}", k + 1
        labels.append(label)

    df, cols = None, []
    for r, label in zip(results, labels):
        e_col, s_col = f"E_{label}_MeV", f"dEdx_{label}{suffix}"
        cols += [e_col, s_col]
        d = pd.DataFrame({
            x_label: r.x_in(mass_thickness),
            e_col:   r.E,
            s_col:   r.dEdx_in(unit),
        })
        df = d if df is None else pd.merge(df, d, on=x_label, how="outer")

    df = df.sort_values(x_label).reset_index(drop=True)
    filled = df.set_index(x_label)[cols].interpolate(method="index", limit_area="inside")
    df[cols] = filled.to_numpy()
    return df


def export_csv(results: List[TrackResult], path: str,
               unit: StoppingUnit, mass_thickness: bool = False) -> None:
    _build_export_df(results, unit, mass_thickness).to_csv(path, index=False)


def _write_sheet(ws, df: pd.DataFrame, max_width: int, freeze_header: bool = False) -> None:
    """Write df to a worksheet with a styled header row and auto-sized columns"""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows

    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="1E3A5F")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center")

    for col in ws.columns:
        longest = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(longest + 2, max_width)

    if freeze_header:
        ws.freeze_panes = "A2"


_STOP_TEXT = {
    "stopped": "particle stopped",
    "model_limit": "electronic-stopping model limit",
    "x_stop": "depth limit (range NOT reached)",
    "max_steps": "step limit (range NOT reached)",
}


def export_xlsx(results: List[TrackResult], path: str,
                unit: StoppingUnit, mass_thickness: bool = False) -> None:
    """
    Sheet "Bragg curves"
        One shared x-column, then alternating energy / dE/dx columns per
        particle — same structure as the CSV export so the two are
        interchangeable (the Excel headers additionally carry the unit)

    Sheet "Metadata"
        Entry/final energy, why the track ended, range, peak dE/dx and
        unit information for each particle
    """
    import openpyxl

    df = _build_export_df(results, unit, mass_thickness, unit_in_name=True)

    df_meta = pd.DataFrame([{
        "Particle": r.name,
        "Entry energy (MeV)": float(f"{r.E[0]:.6g}"),
        "Final energy (MeV)": float(f"{r.E[-1]:.6g}"),
        "Track end": _STOP_TEXT[r.stop_reason],
        "Range (mm)": round(r.range_mm, 2),
        # significant digits, not decimals: gas ranges are ~1e-4 g/cm²
        "Range (g/cm²)": float(f"{r.range_mass:.6g}"),
        f"Peak dE/dx ({unit.label})": round(r.peak_dEdx(unit), 5),
        "dE/dx unit": unit.label,
        "x-axis": "Mass thickness (g/cm²)" if mass_thickness else "Distance (mm)",
    } for r in results])

    wb = openpyxl.Workbook()
    ws_data = wb.active
    ws_data.title = "Bragg curves"
    _write_sheet(ws_data, df, max_width=30, freeze_header=True)
    _write_sheet(wb.create_sheet("Metadata"), df_meta, max_width=35)
    wb.save(path)
