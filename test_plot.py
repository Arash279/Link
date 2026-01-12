import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Load simulation data (CSV)
# ==========================
sim = pd.read_csv("Ztotal_result.csv")  # columns: frequency_Hz, Z_magnitude, Z_phase_deg

# ==========================
# Load experiment data (DB)
# ==========================
DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"

conn = sqlite3.connect(DB_PATH)
query = "SELECT Freq, Zabs, Phase FROM exp_10"
exp = pd.read_sql_query(query, conn)
conn.close()

# ==========================
# Sort both by frequency
# ==========================
sim = sim.sort_values("frequency_Hz")
exp = exp.sort_values("Freq")

# ==========================
# Combined plot (magnitude + phase)
# ==========================

plt.figure(figsize=(12, 8))  # 横向拉长

# ----- 上图: log10(|Z|) -----
plt.subplot(2, 1, 1)
plt.semilogx(sim["frequency_Hz"], np.log10(sim["Z_magnitude"]), label="Simulation", linewidth=2)
plt.semilogx(exp["Freq"], np.log10(exp["Zabs"]), label="Experiment (exp_10)", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("log10(|Z|) (Ohm)")
plt.title("Impedance Magnitude Comparison (log scale)")
plt.grid(True)
plt.legend()

# ----- 下图: Phase -----
plt.subplot(2, 1, 2)
plt.semilogx(sim["frequency_Hz"], sim["Z_phase_deg"], label="Simulation", linewidth=2)
plt.semilogx(exp["Freq"], exp["Phase"], label="Experiment (exp_10)", linewidth=2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (deg)")
plt.title("Impedance Phase Comparison")
plt.grid(True)
plt.legend()

plt.tight_layout()  # 自适应排版
plt.show()

