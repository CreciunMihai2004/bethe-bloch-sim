import numpy as np
import pandas as pd
import plotly.graph_objects as go

A = 10
Z1, Z2 = 4, 5
E01, E02 = 300, 400
r = 0.5

x_start = 0
x_stop = 1000
delta_x = 20

def bethe_bloch_simulation(A, Z, E0, name):
    x_vals = [x_start]
    E_vals = [E0]
    dEdx_vals = []
    
    current_x = x_start
    current_E = E0
    
    while current_x < x_stop and current_E > 0:
        dEdx = (-r * Z**2 * A) / current_E
        dEdx_vals.append(dEdx)
        
        current_E += dEdx * delta_x
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

    return pd.DataFrame({
        'x': x_vals, 
        f'E_{name}': E_vals, 
        f'dE/dx_{name}': dEdx_vals
    })

df1 = bethe_bloch_simulation(A, Z1, E01, 'Z1')
df2 = bethe_bloch_simulation(A, Z2, E02, 'Z2')

df = pd.merge(df1, df2, on='x', how='outer')

intersections = []

diff = df['dE/dx_Z1'] - df['dE/dx_Z2']

valid_diff = diff.dropna()

cross_idx = np.where(np.diff(np.sign(valid_diff)) != 0)[0]

if len(cross_idx) > 0:
    
    idx = valid_diff.index[cross_idx[0]]
    
    x_a, x_b = df['x'].iloc[idx], df['x'].iloc[idx + 1]
    y1_a, y1_b = df['dE/dx_Z1'].iloc[idx], df['dE/dx_Z1'].iloc[idx + 1]
    y2_a, y2_b = df['dE/dx_Z2'].iloc[idx], df['dE/dx_Z2'].iloc[idx + 1]
    
    dy1 = y1_b - y1_a
    dy2 = y2_b - y2_a
    
    t = (y2_a - y1_a) / (dy1 - dy2)
    
    int_x = x_a + t * (x_b - x_a)
    int_y = y1_a + t * dy1
    
    intersections.append((int_x, int_y))
    
print(intersections)
       
df.to_csv('bethe_bloch_simulation_2.csv', index = False)
 
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['x'], 
    y=df['dE/dx_Z1'], 
    mode='lines+markers', 
    name=f'Element 1 (Z={Z1}, E0={E01})', 
    line=dict(color='cyan')))

fig.add_trace(go.Scatter(
    x=df['x'], 
    y=df['dE/dx_Z2'], 
    mode='lines+markers', 
    name=f'Element 2 (Z={Z2}, E0={E02})', 
    line=dict(color='orange')))

for (ix, iy) in intersections:
    fig.add_trace(go.Scatter(
        x=[ix], y=[iy], 
        mode='markers', 
        marker=dict(size=14, color='red', symbol='x'),
        name=f'Intersection ({ix:.5f}, {iy:.5f})'
    ))

fig.update_layout(
    title='dE/dx (MeV/m) vs x (mm)',
    xaxis_title='x (mm)',
    yaxis_title='dE/dx (MeV/mm)',
    template='plotly_dark',
)

fig.show()