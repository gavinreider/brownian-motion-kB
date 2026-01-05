from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PICS_DIR = BASE_DIR / "pictures-videos"


# Load counts vs depth
df = pd.read_csv(DATA_DIR / "particle_counts_vs_depth.csv")

# Depth in microns and counts
z_um = df["depth_um"].values
counts = df["count"].values

# Choose depth range where law should hold
mask = (z_um >= 115) & (z_um <= 160) & (counts > 0)
z_sel_um = z_um[mask]
n_sel = counts[mask]

# Convert z to meters
z_sel = z_sel_um * 1e-6


# Fit ln(count) vs z
ln_n = np.log(n_sel)

# Linear fit: ln n = a + s*z    (s is slope)
s, a = np.polyfit(z_sel, ln_n, 1)

print("Slope s =", s, "1/m")

# Plot to visually check linearity
plt.scatter(z_sel_um, ln_n, label="data")
plt.plot(z_sel_um, a + s*z_sel, label="fit")
plt.xlabel("Depth (µm)")
plt.ylabel("ln(count)")
plt.legend()
PICS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(PICS_DIR / "ln_count_fit.png", dpi=200, bbox_inches="tight")
plt.close()

# Computing Boltzmann constant from slope

# Bead and fluid properties 
r = 0.5e-6                   # bead radius (m) (1 micron diameter)
rho_bead = 1050.0            # kg/m^3 (polystyrene ~1.05 g/cm^3)
rho_water = 1000.0           # kg/m^3
delta_rho = rho_bead - rho_water

V = (4.0/3.0) * np.pi * r**3 # bead volume
m_eff = delta_rho * V        # buoyant mass

g = 9.81                     # m/s^2
T = 296.15                    # K (23°C)

k_B = - m_eff * g / (s * T)

print("Effective bead mass m_eff = ", m_eff, "kg")
print("Estimated Boltzmann constant k_B = ", k_B, "J/K")
print("Ratio to accepted value =", k_B / 1.380649e-23)
