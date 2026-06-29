import numpy as np
import pandas as pd
import plotly.graph_objects as go

K = 0.307075              # MeV cm^2 / mol
me_c2 = 0.510998918       # Electron mass * c^2 (MeV)

# # Absorber (Gas) Parameters  (Argon)
Z_gas = 18
A_gas = 39.948           # g/mol 
I_gas = 188.0 * 1e-6     # Mean excitation energy (MeV)
rho_gas = 0.00178        # Density (g/cm^3)   

# Absorber (Gas) Parameters (C4H10)
# Z_gas = 34
# A_gas = 58.124           # g/mol
# I_gas = 48.2 * 1e-6    # Mean excitation energy
# rho_gas = 7.5 * 1e-6            # Density (g/cm^3)

# Incident Particle Parameters
# Beryllium
z_p1 = 4                   # Charge
M_p1 = 9314                # Mass (MeV/c^2)
E01 = 1.8                  # Initial kinetic energy (MeV)

# Boron
z_p2 = 5                   # Charge 
M_p2 = 9314                # Mass (MeV/c^2)
E02 = 1.6                  # Initial kinetic energy (MeV)

# Settings
x_start = 0
x_stop = 1000               # mm
delta_x = 0.01              # mm

def bethe_bloch_simulation(Z_gas, A_gas, I_gas, rho_gas, z, M, E0_k, name):
    x_vals = [x_start]
    E_vals = [E0_k]
    dEdx_vals = []
    
    current_x = x_start
    current_E = E0_k
    
    while current_x < x_stop and current_E > 0.1: 
        
        # Relativistic kinematics (dimentionless)
        gamma = (current_E / M) + 1.0
        beta_sq = 1.0 - (1.0 / gamma**2)
        beta = np.sqrt(beta_sq)
        
        # z_eff = z * (1.0 - np.exp(-125.0 * beta * (z ** (-2/3))))
        
        # T_max (MeV)
        ratio = me_c2 / M
        T_max = (2.0 * me_c2 * beta_sq * gamma**2) / (1.0 + 2.0 * gamma * ratio + ratio**2)
        
        # Logarithm argument (dimentionless)
        log_arg = (2.0 * me_c2 * beta_sq * gamma**2 * T_max) / (I_gas**2)
        # log_arg = (2.0 * me_c2 * beta_sq) / (I_gas * (1 - beta_sq))
        
        # Bethe-Bloch Mass Stopping Power (MeV g-1 cm^2) 
        mass_stopping = K * (z**2) * (Z_gas / A_gas) * (1.0 / beta_sq) * (0.5 * np.log(log_arg) - beta_sq)
        # mass_stopping = K * (z**2) * (Z_gas / A_gas) * (1.0 / beta_sq) * (np.log(log_arg) - beta_sq)
        
        # Linear Stopping Power (MeV / cm)
        linear_stopping = mass_stopping * rho_gas
        
        # Convert to MeV / mm
        dEdx_mm = linear_stopping / 100
        
        dEdx_vals.append(dEdx_mm)
        
        # Subtract energy loss
        current_E -= dEdx_mm * delta_x
        current_x += delta_x

        if current_E <= 0:
            current_E = 0
            x_vals.append(current_x)
            E_vals.append(current_E)
            break
            
        x_vals.append(current_x)
        E_vals.append(current_E)
        
    if len(dEdx_vals) < len(x_vals):
        dEdx_vals.append(np.nan)

    df = pd.DataFrame({
        'x': x_vals, 
        f'E_{name}': E_vals, 
        f'dE/dx_{name}': dEdx_vals
    })
    
    return df

# Run the simulations
df1 = bethe_bloch_simulation(Z_gas, A_gas, I_gas, rho_gas, z_p1, M_p1, E01, '1')
df2 = bethe_bloch_simulation(Z_gas, A_gas, I_gas, rho_gas, z_p2, M_p2, E02, '2')

df = pd.merge(df1, df2, on='x', how='outer')

eps = 0.00001
df.loc[df['dE/dx_1'] < eps, 'dE/dx_1'] = np.nan
df.loc[df['dE/dx_2'] < eps, 'dE/dx_2'] = np.nan

df = df.dropna(subset=['dE/dx_1', 'dE/dx_2'], how='all')

df.to_csv('bethe_bloch_simulation.csv', index = False)

# Intersection calculation
intersections = []
diff = df['dE/dx_1'] - df['dE/dx_2']
valid_diff = diff.dropna()
cross_idx = np.where(np.diff(np.sign(valid_diff)) != 0)[0]

if len(cross_idx) > 0:
    idx = valid_diff.index[cross_idx[0]]
    
    x_a, x_b = df['x'].iloc[idx], df['x'].iloc[idx + 1]
    y1_a, y1_b = df['dE/dx_1'].iloc[idx], df['dE/dx_1'].iloc[idx + 1]
    y2_a, y2_b = df['dE/dx_2'].iloc[idx], df['dE/dx_2'].iloc[idx + 1]
    
    dy1 = y1_b - y1_a
    dy2 = y2_b - y2_a
    
    t = (y2_a - y1_a) / (dy1 - dy2)
    
    int_x = x_a + t * (x_b - x_a)
    int_y = y1_a + t * dy1
    
    intersections.append((int_x, int_y))
    
# print(intersections)

# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['x'], 
    y=df['dE/dx_1'], 
    mode='lines', 
    name=f'Beryllium (Z={z_p1})', 
    line=dict(color='cyan')))

fig.add_trace(go.Scatter(
    x=df['x'], 
    y=df['dE/dx_2'], 
    mode='lines', 
    name=f'Boron (Z={z_p2})', 
    line=dict(color='orange')))

for (ix, iy) in intersections:
    fig.add_trace(go.Scatter(
        x=[ix], y=[iy], 
        mode='markers', 
        marker=dict(size=14, color='red', symbol='x'),
        name=f'Intersection ({ix:.2f}, {iy:.2f})'
    ))

fig.update_layout(
    title='Bragg Curve: dE/dx (MeV/mm) vs x (mm)',
    xaxis_title='x (mm)',
    yaxis_title='dE/dx (MeV/mm)',
    template='plotly_dark',
)

fig.show()