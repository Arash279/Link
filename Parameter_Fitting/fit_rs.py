#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates Rs (DC Stator Resistance) using Low-Frequency Extrapolation.

This script addresses the core problem: the available data is from a "turning"
(no-load, s~0) condition, not a locked-rotor (s=1) condition.

Strategy:
1.  As per the paper1 definition, Rs is the DC resistance.
2.  In a no-load (s~0) model, the rotor branch is open.
3.  The total impedance Z(f) approaches Rs as frequency f -> 0.
4.  We load all available impedance data (exp_7, 8, 9).
5.  We fit a line to the REAL part of the impedance vs. frequency.
6.  The y-intercept (at f=0) of this line is the estimated DC Rs.

This avoids all model-mismatch errors from previous fittings.
"""

import os, sqlite3
import numpy as np
from scipy.stats import linregress
from typing import Tuple, List

# ======== USER INPUTS / CONFIGURATION ========
DB_PATH = r"D:\Desktop\data\AP_1p5.db"
TABLES = ["exp_7", "exp_8", "exp_9"]

# Frequency range to use for the extrapolation fit
F_FIT_MIN = 100.0
F_FIT_MAX = 500.0


# --------- Data Loader ----------
def load_table(db_path: str, table: str, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Loads Freq and Z (complex) from the database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
            rows = cur.fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error in table '{table}': {e}")
    if not rows: raise ValueError(f"No rows in table '{table}'")

    arr = np.array(rows, dtype=float)
    f, zabs, ph_deg = arr[:, 0], arr[:, 1], arr[:, 2]
    mask = (f >= fmin) & (f <= fmax)
    f, zabs, ph_deg = f[mask], zabs[mask], ph_deg[mask]

    if len(f) == 0:
        raise ValueError(f"No samples in [{fmin},{fmax}] Hz for table '{table}'")

    ph_rad = np.deg2rad(ph_deg)
    Z = zabs * np.exp(1j * ph_rad)
    return f, Z


# --------- Main Extrapolation Logic ----------
def main():
    print("=" * 80)
    print("Estimating Rs (DC Stator Resistance) via Low-Frequency Extrapolation")
    print(f"Database: {DB_PATH}")
    print(f"Tables: {', '.join(TABLES)}")
    print(f"Frequency Range: [{F_FIT_MIN:.0f}, {F_FIT_MAX:.0f}] Hz")
    print("-" * 80)

    all_freqs = []
    all_real_z = []

    try:
        # 1. Load data from all tables
        for table in TABLES:
            f, Z = load_table(DB_PATH, table, F_FIT_MIN, F_FIT_MAX)
            all_freqs.append(f)
            all_real_z.append(Z.real)
            print(f"  Loaded {len(f)} data points from {table}")

        # 2. Combine all data into two large arrays
        f_data = np.concatenate(all_freqs)
        r_data = np.concatenate(all_real_z)
        print(f"  Total data points for fit: {len(f_data)}")

        # 3. Perform linear regression: R(f) = (slope * f) + intercept
        #    We are looking for the intercept, which is R(0) = Rs
        fit = linregress(f_data, r_data)

        Rs_estimated = fit.intercept

        print("\n" + "-" * 80)
        print("Fit Results (R = slope * f + intercept):")
        print(f"  Slope:     {fit.slope:.6e} (Ohm/Hz)")
        print(f"  R-squared: {fit.rvalue ** 2:.6f}")

        print("\n" + "=" * 80)
        print(f"  ESTIMATED Rs (DC) = Intercept @ f=0")
        print(f"  Rs ≈ {Rs_estimated:.6f} Ω")
        print("=" * 80)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        import sys
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
