# BraggSim: Particle Energy Loss & Bragg Curve Simulator

BraggSim calculates continuous slowing down approximation ranges, stopping powers, and plots Bragg curves, combining multiple physics models to ensure accuracy from high energies all the way down to a complete stop.

## Features
*   **Interactive UI:** A clean Qt-based interface for fast parameter adjustments and real-time plotting.
*   **Extensible Databases:** Add new materials and projectiles simply by editing external CSV files—no recompilation required.
*   **Robust Physics Engine:** Smoothly transitions between high-energy and low-energy stopping power models.
*   **Standalone Deployment:** Available as a single, portable Windows executable (`.exe`).

## 📂 Project Structure

To ensure the application runs correctly—either from source or as a compiled executable—the `data/` directory must be kept alongside the script or `.exe`:

```text
BraggSim/
│
├── main.py                # Main application entry point
└── core/                  # Core integration and stopping power models
    ├── constants.py
    ├── materials.py
    ├── particles.py
    ├── physics.py
    ├── simulation.py
    └── units.py
└── ui/                    # User interface 
    ├── main_window.py
    └── plot_widget.py
│
└── data/                  # External configuration files
    ├── materials.csv
    └── particles.csv
```

## The Physics Models
The simulation engine relies on a hybrid model to ensure numerical stability and physical accuracy across the entire energy spectrum:

*   **High-Energy Electronic Stopping:** Uses the standard **Bethe-Bloch formula**, complete with Sternheimer density-effect corrections.
*   **Nuclear Stopping:** Integrates the **ZBL (Ziegler-Biersack-Littmark)** universal potential model to account for elastic collisions with target nuclei, which dominates at the very end of the particle's track.

## Customizing the Data (CSVs)
You can add custom target materials or projectile particles at any time. Simply open the data/ folder and edit the CSV files.

materials.csv
Defines the target materials. Required columns:

**name**: Material identifier (e.g., Water, Silicon)

**Z**: Atomic number

**A**: Atomic mass (g/mol)

**I_eV**: Mean excitation energy (eV)

**rho**: Density (g/cm³)

(Optional) **Sternheimer parameters**: X0, X1, a_stern, m_stern

particles.csv
Defines the incident projectiles. Required columns:

**name**: Particle identifier (e.g., Proton, Alpha)

**z**: Charge number

**M_u**: Rest mass in atomic mass units (amu)

(Optional) **E0**: Default initial kinetic energy (MeV)