"""
Simulation and post-processing
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .materials import Material
from .particles import Particle
from .physics import dEdx_mass
from .units import StoppingUnit, convert_from_mass

_MAX_STEPS = 500_000       # safety cap on the number of integration steps
_PROGRESS_EVERY = 0.002    # report progress every 0.2 % of the energy lost
_MIN_STEP_FACTOR = 1e-4    # smallest allowed step, as a fraction of max_dx
_TRIM_FRACTION = 0.015     # see _trim_after_peak


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
    Bragg curve result.  Arrays are already trimmed to the physical
    range — they end where dE/dx first drops below 1.5 % of the Bragg
    peak after the peak, which marks the boundary of Bethe-Bloch
    validity (nuclear stopping takes over below that point)

    x_mass    : mass thickness (g/cm^2)
    E         : kinetic energy (MeV)
    dEdx_mass : mass stopping power (MeV cm^2 / g)
    rho       : material density (g/cm^3)
    """
    name: str
    color: str
    x_mass: np.ndarray
    E: np.ndarray
    dEdx_mass: np.ndarray
    rho: float

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


# ---- cutoff helper ----

def _trim_after_peak(x_vals: list, E_vals: list, dEdx_vals: list,
                     threshold_frac: float = _TRIM_FRACTION):
    """Cut all three per-step lists where dE/dx falls below threshold_frac of its peak"""

    arr = np.array(dEdx_vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return x_vals, E_vals, dEdx_vals

    peak      = float(np.nanmax(finite))
    threshold = threshold_frac * peak
    peak_idx  = int(np.nanargmax(arr))

    post  = arr[peak_idx:]
    below = np.where(np.isfinite(post) & (post < threshold))[0]

    if len(below) == 0:
        return x_vals, E_vals, dEdx_vals

    cutoff = peak_idx + int(below[0])
    return x_vals[:cutoff], E_vals[:cutoff], list(arr[:cutoff])


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

    while x < x_stop_m and E > settings.E_cutoff and steps < _MAX_STEPS:
        steps += 1
        s1 = dEdx_mass(mat, part, E)
        if s1 <= 0:
            break

        dx = max(min(max_dx_m, settings.frac_loss * E / s1), min_dx_m)

        # midpoint rule: use the stopping power half-way through the step
        E_mid = E - 0.5 * dx * s1
        s_mid = dEdx_mass(mat, part, E_mid) if E_mid > 0 else 0.0
        if s_mid <= 0:
            # the particle stops inside this step: record it with the entry value
            dEdx_vals.append(s1)
            x += dx
            x_vals.append(x)
            E_vals.append(0.0)
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

    # every x has one dE/dx except the final point
    if len(dEdx_vals) < len(x_vals):
        dEdx_vals.append(np.nan)

    if progress_cb is not None:
        progress_cb(1.0)

    x_vals, E_vals, dEdx_vals = _trim_after_peak(x_vals, E_vals, dEdx_vals)

    return TrackResult(
        name=part.name,
        color=part.color,
        x_mass=np.array(x_vals),
        E=np.array(E_vals),
        dEdx_mass=np.array(dEdx_vals),
        rho=mat.rho,
    )


# ---- intersection finder ----

def find_intersections(a: TrackResult, b: TrackResult,
                       unit: StoppingUnit,
                       mass_thickness: bool = False) -> List[tuple]:
    xa, ya = a.x_in(mass_thickness), a.dEdx_in(unit)
    xb, yb = b.x_in(mass_thickness), b.dEdx_in(unit)

    df_a = pd.DataFrame({"x": xa, "ya": ya})
    df_b = pd.DataFrame({"x": xb, "yb": yb})
    df = (pd.merge(df_a, df_b, on="x", how="outer")
            .sort_values("x").reset_index(drop=True))

    # Fill each curve on the merged grid by x value.  A plain interpolate()
    # works by row position, which is wrong when the two curves have
    # different step sizes (up to ~5 % off in a dense material)
    by_x = df.set_index("x")
    df["ya"] = by_x["ya"].interpolate(method="index", limit_area="inside").to_numpy()
    df["yb"] = by_x["yb"].interpolate(method="index", limit_area="inside").to_numpy()

    diff = (df["ya"] - df["yb"]).dropna()
    # Compare "a >= b" instead of sign(): a difference of exactly 0 at a grid
    # point (sign -1, 0, +1) would otherwise count as two crossings
    nonneg    = (diff >= 0).to_numpy()
    cross_idx = np.where(nonneg[:-1] != nonneg[1:])[0]

    intersections = []
    for loc in cross_idx:
        idx = diff.index[loc]
        nxt = diff.index[loc + 1]
        x_a, x_b = df.loc[idx, "x"],  df.loc[nxt, "x"]
        y1a, y1b = df.loc[idx, "ya"], df.loc[nxt, "ya"]
        y2a, y2b = df.loc[idx, "yb"], df.loc[nxt, "yb"]
        dy1, dy2 = y1b - y1a, y2b - y2a
        if (dy1 - dy2) != 0:
            t  = (y2a - y1a) / (dy1 - dy2)
            ix = x_a + t * (x_b - x_a)
            iy = y1a + t * dy1
            intersections.append((ix, iy))

    # a curve that only touches the other at a grid point is reported once
    unique: List[tuple] = []
    for ix, iy in intersections:
        if not unique or not np.isclose(ix, unique[-1][0], rtol=1e-9, atol=0.0):
            unique.append((ix, iy))
    return unique


# ---- export ----

def _build_export_df(results: List[TrackResult], unit: StoppingUnit,
                     mass_thickness: bool, unit_in_name: bool = False) -> pd.DataFrame:
    """
    Shared table-building logic for CSV and Excel export

    All particles share one x column.  Both the energy and the dE/dx
    columns are filled in by x value between a particle's own points, and
    left empty beyond its range.  Two particles with the same name get a
    "#2" suffix so their columns do not collide
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


def export_xlsx(results: List[TrackResult], path: str,
                unit: StoppingUnit, mass_thickness: bool = False) -> None:
    """
    Sheet "Bragg curves"
        One shared x-column, then alternating energy / dE/dx columns per
        particle — same structure as the CSV export so the two are
        interchangeable (the Excel headers additionally carry the unit)

    Sheet "Metadata"
        Range, peak dE/dx, and unit information for each particle
    """
    import openpyxl

    df = _build_export_df(results, unit, mass_thickness, unit_in_name=True)

    df_meta = pd.DataFrame([{
        "Particle": r.name,
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
