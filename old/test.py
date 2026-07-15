import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

# 1. Definire functii
def f(x, y):
    return x**2 + y

def g(x, y):
    return 3*x + 2*y

# 2. Definire domenii + vectori
x_start, x_stop, x_step = 0, 5, 0.1
# y_start, y_stop, y_step = 1, 3, 1

x_vals = np.arange(x_start, x_stop + x_step, x_step)
y_vals = [1, 2, 5]

# Initializare liste pentru stocarea datelor
all_data = []
intersection_data = []

# Setup plot
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

for idx, y in enumerate(y_vals):
    color = colors[idx % len(colors)]
    
    # Generare valori pentru f si g
    z1 = f(x_vals, y)
    z2 = g(x_vals, y)
    
    # Plotare functii
    ax.plot(x_vals, z1, label=f'$f(x, y={y})$', color=color, linestyle='-')
    ax.plot(x_vals, z2, label=f'$g(x, y={y})$', color=color, linestyle='--')
    
    # Stocare valori in dictionar
    for x, val1, val2 in zip(x_vals, z1, z2):
        all_data.append({'x': x, 'y': y, 'f(x,y)': val1, 'g(x,y)': val2})
        
    # Determinare intersectie numeric
    # f(x,y) - g(x,y) = 0
    def target_equation(x_val):
        return f(x_val, y) - g(x_val, y)
    
    # Detectare schimbari de semn pt. a identifica intervale cu posibile radacini
    diff = z1 - z2
    found_roots = []
    for i in range(len(x_vals) - 1):
        if diff[i] * diff[i+1] <= 0:
            initial_guess = (x_vals[i] + x_vals[i+1]) / 2
            root, info, ier, msg = fsolve(target_equation, initial_guess, full_output=True)
            if ier == 1:
                root_x = root[0]
                if x_start <= root_x <= x_stop:
                    # Verificare daca radacina a fost deja gasita
                    if not any(np.isclose(root_x, r, atol=1e-3) for r in found_roots):
                        found_roots.append(root_x)
                        
    # Plotare puncte de intersectie + stocare
    for root_x in found_roots:
        root_z = f(root_x, y)
        ax.plot(root_x, root_z, 'ro', markersize=8, 
                label='Intersectie' if idx == 0 and root_x == found_roots[0] else "")
        intersection_data.append({'y': y, 'intersection_x': root_x, 'intersection_z': root_z})

# Formatare ploturi
ax.set_title('Intersectia $f(x,y)$ si $g(x,y)$', fontsize=14)
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x,y)$', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('functions_intersection.png')

# Exportare date
pd.DataFrame(all_data).to_csv('all_function_values.csv', index=False)
pd.DataFrame(intersection_data).to_csv('intersection_points.csv', index=False)
