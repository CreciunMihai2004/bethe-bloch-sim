import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Input parametri
A = 10
Z = 4
# Z2 = 5
E0 = 300
# E02 = 350
r = 0.5

x_start = 0
x_stop = 600
delta_x = 20

# Iteratii
x_vals = [x_start]
E_vals = [E0]
dEdx_vals = []

current_x = x_start
current_E = E0

while current_x < x_stop and current_E > 0:
    # Calculare dE/dx = (-r * Z^2 * A) / E_i
    dEdx = (-r * Z**2 * A) / current_E
    dEdx_vals.append(dEdx)
    
    # Actualizare energie si pozitie
    current_E += dEdx * delta_x
    current_x += delta_x
    
    if current_E < 0:
        current_E = 0
    
    x_vals.append(current_x)
    E_vals.append(current_E)
    
if len(dEdx_vals) < len(x_vals):
    dEdx_vals.append(np.nan)
    
# Stocare in CSV
df = pd.DataFrame({
    'x': x_vals,
    'E': E_vals,
    'dE/dx': dEdx_vals
})

df.to_csv('bethe_bloch_simulation.csv', index = False)

# Plotare
fig = px.line(
    df, 
    x='x', 
    y='dE/dx', 
    markers=True,
    title='dE/dx / x',
    labels={
        'x': 'x (mm)', 
        'dE/dx': 'dE/dx'
    }
)

fig.update_layout(
    xaxis_title="x (mm)",
    yaxis_title="dE/dx (MeV/mm)",
    template="plotly_dark",
)

fig.show()
fig.write_image("bethe_bloch_simulation.png", height=900, width=1600, scale=2)