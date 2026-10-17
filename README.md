# BraggSim: Particle Energy Loss & Bragg Curve Simulator

BraggSim calculates stopping powers and ranges in the continuous slowing-down approximation (CSDA) and plots Bragg curves for charged particles in gases and solids.

## Features
*   **Interactive UI:** A clean Qt-based interface for fast parameter adjustments and real-time plotting.
*   **Extensible Databases:** Add new materials and projectiles simply by editing CSV files.
*   **Honest limits:** the track ends where the electronic-stopping model stops being valid, and the app tells you how much energy is left untracked.
*   **Tested:** `pytest` suite checks the physics against NIST PSTAR and published ZBL properties.
*   **Standalone Deployment:** Available as a single, portable Windows executable (`.exe`).

## 📂 Project Structure

```text
BraggSim/
│
├── main.py                # Main application entry point
├── tests/                 # pytest suite (run: python -m pytest)
├── docs/PROGRESS.md       # milestones, validation log, open questions
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

*   **Electronic stopping:** the **Bethe-Bloch formula** with a velocity-dependent effective charge (Barkas form) and the Sternheimer density-effect correction.
*   **Nuclear stopping:** the **ZBL (Ziegler-Biersack-Littmark)** universal potential, for elastic collisions with target nuclei.

### Known limits (read before quoting a range)

*   Bethe-Bloch is not valid at low velocity. When its logarithm falls to about 1 the electronic stopping is switched off, and the track **ends there** (`stop_reason = "model_limit"`). For Be-10 in isobutane that is about 0.22 MeV. The slowing-down below it is **not** in the reported range; the results line shows how much energy is left untracked.
*   There are no shell, Barkas or Bloch corrections yet, so agreement with reference data is a few percent at about 1 MeV for protons and worse for heavy ions.
*   Compounds are treated as one atom with an effective Z (nuclear stopping uses that Z too).

## Tests

```text
pip install -r requirements-dev.txt
python -m pytest
```

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

(Optional) **E0**: Default initial kinetic energy (MeV); leave blank to enter it in the UI
