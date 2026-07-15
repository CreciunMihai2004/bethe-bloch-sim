"""
Bragg curve simulator based on the Bethe-Bloch stopping power formula,
extended with the corrections needed to get reasonable agreement with
semi-empirical references such as SRIM/PSTAR/ASTAR in the few-MeV range:

  - effective (screened) projectile charge   z_eff(beta, z)
  - Barkas/ICRU shell correction             C(I, eta)/Z
  - Sternheimer density-effect correction    delta(beta*gamma)
  - adaptive-step RK2 integration in x, instead of fixed-step Euler

Why this matters physically:
  Plain Bethe-Bloch assumes a fully-stripped projectile and orbital
  electrons that are always much faster than the projectile. Both
  assumptions break down for a few-MeV proton or alpha particle, which
  is exactly the Bragg-peak region. SRIM and the NIST PSTAR/ASTAR
  tables handle this with empirical low-energy corrections; the
  corrections below are the analytic approximations to the same effects.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# 1. Universal physical constants
# ============================================================
K = 0.307075           # MeV cm^2 / mol
me_c2 = 0.510998918    # electron rest energy (MeV)

# ============================================================
# 2. Detector gas: Isobutane (C4H10)
# ============================================================
Z_gas = 34.0                       # electrons per molecule
A_gas = 58.124                     # g/mol
I_gas = 48.3 * 1e-6                    # MeV (isobutane ~=48.3eV)
rho_gas = 0.00125                  # g/cm^3

I_eV = I_gas * 1e6                 # I in eV, needed for the shell-correction formula

# Plasma energy of the medium (Sternheimer), in eV
hbar_wp_eV = 28.816 * np.sqrt(rho_gas * Z_gas / A_gas)
Cbar = 2.0 * np.log(I_eV / hbar_wp_eV) + 1.0

# Generic Sternheimer parameters for a low-density gas (no conduction electrons).
# These vary slightly by material; for a light hydrocarbon gas at STP the
# density effect only becomes non-negligible above beta*gamma ~ a few, far
# above the velocities reached by few-MeV protons/alphas, so these defaults
# are adequate here.
X0_gas, X1_gas, a_stern, m_stern = 1.6, 4.0, 0.10, 3.0


def density_effect(beta, gamma):
    """Sternheimer density-effect correction delta(X), X = log10(beta*gamma)."""
    bg = beta * gamma
    if bg <= 0:
        return 0.0
    X = np.log10(bg)
    if X < X0_gas:
        return 0.0
    elif X < X1_gas:
        return 4.6052 * X - Cbar + a_stern * (X1_gas - X) ** m_stern
    else:
        return 4.6052 * X - Cbar


def shell_correction(eta):
    """
    Barkas/ICRU37 shell correction C(I, eta)/Z.
    eta = beta*gamma (projectile reduced momentum).
    Formula is only valid for eta >~ 0.13 (roughly E > a few MeV/nucleon
    for protons); below that we switch it off rather than extrapolate
    into a regime where it is known to be unreliable.
    """
    if eta < 0.13:
        return 0.0
    eta2 = eta ** 2
    eta4 = eta2 ** 2
    eta6 = eta2 ** 3
    term1 = (0.422377 / eta2 + 0.0304043 / eta4 - 0.00038106 / eta6) * 1e-6 * I_eV ** 2
    term2 = (3.850190 / eta2 - 0.1667989 / eta4 + 0.00157955 / eta6) * 1e-9 * I_eV ** 3
    C = term1 + term2
    return max(C, 0.0)


def effective_charge(z, beta):
    """
    Ziegler/Brandt-Kitagawa-type effective charge for an ion moving through
    matter: the projectile is partially neutralized (electron pickup) at
    low velocity, so the charge the medium 'sees' drops below the bare
    nuclear charge z. This is the dominant low-energy correction for
    alpha particles in particular, and it also regularizes the unphysical
    1/beta^2 divergence of naive Bethe-Bloch as beta -> 0.
    """
    if beta <= 0:
        return 0.0
    return z * (1.0 - np.exp(-125.0 * beta * z ** (-2.0 / 3.0)))


def dEdx_bethe_bloch(Z_target, A_target, I_target, rho_target, z_part, M_part, E_kin):
    """
    Returns linear stopping power in MeV/mm for a projectile of kinetic
    energy E_kin (MeV), including effective charge, shell, and density
    corrections.
    """
    gamma = (E_kin / M_part) + 1.0
    beta_sq = 1.0 - (1.0 / gamma ** 2)
    beta_sq = max(beta_sq, 1e-12)
    beta = np.sqrt(beta_sq)

    z_eff = effective_charge(z_part, beta)
    if z_eff <= 0:
        return 0.0

    ratio = me_c2 / M_part
    T_max = (2.0 * me_c2 * beta_sq * gamma ** 2) / (1.0 + 2.0 * gamma * ratio + ratio ** 2)

    log_arg = (2.0 * me_c2 * beta_sq * gamma ** 2 * T_max) / (I_target ** 2)
    if log_arg <= 1.0:
        return 0.0

    eta = beta * gamma
    C_over_Z = shell_correction(eta) / Z_target
    delta = density_effect(beta, gamma)

    mass_stopping = K * (z_eff ** 2) * (Z_target / A_target) * (1.0 / beta_sq) * (
        0.5 * np.log(log_arg) - beta_sq - delta / 2.0 - C_over_Z
    )
    mass_stopping = max(mass_stopping, 0.0)

    # linear_stopping = mass_stopping * rho_target   # MeV/cm
    linear_stopping = mass_stopping                # MeV g-1 cm2
    return linear_stopping / 100.0                  # MeV/mm


# ============================================================
# 3. Incident particles
# ============================================================
z_p1, M_p1, E0_p1 = 4, 9314, 1.8
z_p2, M_p2, E0_p2 = 5, 9314, 1.6  

# ============================================================
# 4. Step
# ============================================================
x_start = 0.0
x_stop = 1000.0       # mm
E_cutoff = 0.001     # MeV, stop the track here
max_dx = 0.5         # mm, never take a step larger than this
frac_loss = 0.01     # limit energy loss per step to ~1% of current energy


def simulate(Z_t, A_t, I_t, rho_t, z_part, M_part, E0, name):
    x, E = x_start, E0
    x_vals, E_vals, dEdx_vals = [x], [E], []

    while x < x_stop and E > E_cutoff:
        dEdx1 = dEdx_bethe_bloch(Z_t, A_t, I_t, rho_t, z_part, M_part, E)
        if dEdx1 <= 0:
            break

        # adaptive step: smaller steps where dE/dx is large (near the Bragg peak)
        dx = min(max_dx, frac_loss * E / dEdx1)
        dx = max(dx, 1e-5)

        # RK2 (midpoint) step
        E_mid = E - 0.5 * dx * dEdx1
        if E_mid <= 0:
            dEdx_vals.append(dEdx1)
            x += dx
            E = 0.0
            x_vals.append(x)
            E_vals.append(E)
            break

        dEdx_mid = dEdx_bethe_bloch(Z_t, A_t, I_t, rho_t, z_part, M_part, E_mid)

        if dEdx_mid <= 0:
            # The midpoint already lies past the validity cutoff of the
            # stopping-power formula (effectively zero residual range left).
            # Deposit the remaining energy here instead of freezing E while
            # x keeps advancing (which would silently run the track out to
            # x_stop with a flat, unphysical tail).
            dEdx_vals.append(dEdx1)
            x += dx
            E = 0.0
            x_vals.append(x)
            E_vals.append(E)
            break

        E_next = E - dx * dEdx_mid

        dEdx_vals.append(dEdx_mid)
        x += dx
        E = max(E_next, 0.0)
        x_vals.append(x)
        E_vals.append(E)

    if len(dEdx_vals) < len(x_vals):
        dEdx_vals.append(np.nan)

    return pd.DataFrame({
        "x": x_vals,
        f"E_{name}": E_vals,
        f"dE/dx_{name}": dEdx_vals,
    })


# ============================================================
# 5. Run simulations
# ============================================================
df1 = simulate(Z_gas, A_gas, I_gas, rho_gas, z_p1, M_p1, E0_p1, "Particle1")
df2 = simulate(Z_gas, A_gas, I_gas, rho_gas, z_p2, M_p2, E0_p2, "Particle2")

print(f"Particle1 range:  {df1['x'].iloc[-1]:.4f} mm, "
      f"peak dE/dx = {np.nanmax(df1['dE/dx_Particle1']):.4f} MeV/mm")
print(f"Particle2 range: {df2['x'].iloc[-1]:.4f} mm, "
      f"peak dE/dx = {np.nanmax(df2['dE/dx_Particle2']):.4f} MeV/mm")

df = pd.merge(df1, df2, on="x", how="outer").sort_values("x").reset_index(drop=True)
df["dE/dx_Particle1"] = df["dE/dx_Particle1"].interpolate(limit_area="inside")
df["dE/dx_Particle2"] = df["dE/dx_Particle2"].interpolate(limit_area="inside")

df.to_csv('bethe_bloch_app.csv', index = False)

# ============================================================
# 6. Intersection logic (unchanged approach, now on cleaner data)
# ============================================================
intersections = []
diff = df["dE/dx_Particle1"] - df["dE/dx_Particle2"]
valid_diff = diff.dropna()
cross_idx = np.where(np.diff(np.sign(valid_diff)) != 0)[0]

for idx_loc in cross_idx:
    idx = valid_diff.index[idx_loc]
    try:
        next_idx = valid_diff.index[idx_loc + 1]
    except IndexError:
        continue

    x_a, x_b = df.loc[idx, "x"], df.loc[next_idx, "x"]
    y1_a, y1_b = df.loc[idx, "dE/dx_Particle1"], df.loc[next_idx, "dE/dx_Particle1"]
    y2_a, y2_b = df.loc[idx, "dE/dx_Particle2"], df.loc[next_idx, "dE/dx_Particle2"]

    dy1, dy2 = y1_b - y1_a, y2_b - y2_a
    if (dy1 - dy2) != 0:
        t = (y2_a - y1_a) / (dy1 - dy2)
        int_x = x_a + t * (x_b - x_a)
        int_y = y1_a + t * dy1
        intersections.append((int_x, int_y))

print(f"Intersection points found at: {intersections}")

# ============================================================
# 7. Plotting
# ============================================================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["x"], y=df["dE/dx_Particle1"], mode="lines",
    name=f"Particle 1 - alpha (z={z_p1}, M={M_p1:.0f}, E0={E0_p1} MeV)",
    line=dict(color="cyan", width=3)))

fig.add_trace(go.Scatter(
    x=df["x"], y=df["dE/dx_Particle2"], mode="lines",
    name=f"Particle 2 - proton (z={z_p2}, M={M_p2:.0f}, E0={E0_p2} MeV)",
    line=dict(color="orange", width=3)))

for (ix, iy) in intersections:
    fig.add_trace(go.Scatter(
        x=[ix], y=[iy], mode="markers",
        marker=dict(size=10, color="red", symbol="x"),
        name=f"Intersection ({ix:.2f} mm, {iy:.4f} MeV/mm)"
    ))

fig.update_layout(
    title="Bragg Curves in Isobutane (C4H10) - with effective charge, shell & density corrections",
    xaxis_title="Distance x (mm)",
    yaxis_title="Energy Loss dE/dx (MeV/mm)",
    template="plotly_dark",
    hovermode="x unified",
)

fig.show()
