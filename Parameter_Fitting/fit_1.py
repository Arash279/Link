#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final fitting strategy: Fit for a single, global Rs value only.

- Based on previous results, Rr is assumed to be stable and is now fixed to the
  average of the values from the hybrid fit (~22.66 Ohm).
- All other parameters (Lls, Llr, Lm, Rcore) are also fixed constants.
- The optimizer's only task is to find the single best Rs that minimizes the
  total error across all three experiments (exp_7, exp_8, exp_9).
"""

import os, sqlite3, math
from typing import Tuple, List, Dict
import numpy as np

# SciPy is required for this fitting
try:
    from scipy.optimize import least_squares
except ImportError:
    raise RuntimeError("Please install SciPy: pip install scipy")

# ======== USER INPUTS / CONFIGURATION ========
DB_PATH = r"D:\Desktop\data\AP_1p5.db"
TABLES = ["exp_7", "exp_8", "exp_9"]

# --- Fixed motor constants (All parameters except Rs are now fixed) ---
Lls = 2.55e-2  # H  (Stator leakage inductance)
Llr = 2.55e-2  # H  (Rotor leakage inductance)
Lm = 2.55e-1  # H  (Magnetizing inductance)
Rcore = 4751.342  # Ohm (Set INCLUDE_RCORE=False to ignore)
INCLUDE_RCORE = True

# Rr is now fixed to the average of previous fits.
RR_FIXED = 22.66  # Ohm <--- NEW FIXED PARAMETER

# Fitting frequency window
F_FIT_MIN = 100.0
F_FIT_MAX = 500.0

# --- Parameter Bounds for the ONLY variable being fitted (Rs) ---
RS_MIN, RS_MAX = 0.1, 20.0

# Initial guess for the optimizer
RS0 = 2.0  # Ohm


# --- Helper function to warn if parameters hit their bounds ---
def warn_if_at_bounds(val, lo, hi, name, tol=1e-4):
    """Prints a warning if a value is at its lower or upper bound."""
    if lo != 0 and abs(val - lo) / abs(lo) < tol:
        print(f"  [warn] {name} @ lower bound ({val:.3e} ≈ {lo:.3e})")
    elif lo == 0 and abs(val) < 1e-9:
        print(f"  [warn] {name} @ lower bound ({val:.3e} ≈ {lo:.3e})")
    if abs(val - hi) / abs(hi) < tol:
        print(f"  [warn] {name} @ upper bound ({val:.3e} ≈ {hi:.3e})")


# --------- Data Loader (Unchanged) ----------
def load_table(db_path: str, table: str, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(db_path): raise FileNotFoundError(f"Database not found: {db_path}")
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor();
            cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}");
            rows = cur.fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error in table '{table}': {e}")
    if not rows: raise ValueError(f"No rows in table '{table}'")
    arr = np.array(rows, dtype=float);
    f, zabs, ph_deg = arr[:, 0], arr[:, 1], arr[:, 2]
    mask = (f >= fmin) & (f <= fmax);
    f, zabs, ph_deg = f[mask], zabs[mask], ph_deg[mask]
    if len(f) == 0: raise ValueError(f"No samples in [{fmin},{fmax}] Hz for table '{table}'")
    Z = zabs * np.exp(1j * np.deg2rad(ph_deg))
    return f, Z


# --------- Model (Unchanged) ----------
def Z_branch(omega: np.ndarray, Rs: float, Lls: float, Rr: float, Llr: float, Lm: float, Rcore: float,
             include_Rcore: bool) -> np.ndarray:
    jw = 1j * omega;
    Zs = Rs + jw * Lls;
    Zr = Rr + jw * Llr
    Y = 1.0 / (jw * Lm) + 1.0 / Zr
    if include_Rcore and Rcore > 0: Y += 1.0 / Rcore
    return Zs + (1.0 / Y)


# --------- Fit Only Rs Function ----------
def fit_only_rs(db_path: str, tables: List[str]) -> Dict:
    """Performs a joint fit for a single, global Rs value, with all other parameters fixed."""
    data = []
    for t in tables:
        f, Z = load_table(db_path, t, F_FIT_MIN, F_FIT_MAX)
        w = 1.0 / np.maximum(f, 1.0);
        w /= np.mean(w)  # Frequency weighting
        data.append({"f": f, "w": w, "Z": Z})

    # The parameter vector 'x' now contains only one element: Rs
    x0 = [RS0]
    lb = [RS_MIN]
    ub = [RS_MAX]

    def residual(x: np.ndarray) -> np.ndarray:
        """Computes the weighted residual vector. 'x' is just [Rs]."""
        Rs_fit = x[0]
        res_list = []
        for rec in data:
            omega = 2 * math.pi * rec["f"]
            Z_model = Z_branch(
                omega, Rs=Rs_fit, Lls=Lls, Rr=RR_FIXED, Llr=Llr, Lm=Lm,
                Rcore=Rcore, include_Rcore=INCLUDE_RCORE
            )
            dr = rec["w"] * (Z_model.real - rec["Z"].real)
            di = rec["w"] * (Z_model.imag - rec["Z"].imag)
            res_list.extend([dr, di])
        return np.concatenate(res_list)

    res = least_squares(residual, x0, bounds=(lb, ub), method="trf", loss="soft_l1")
    Rs_final = res.x[0]

    return {
        "Rs": float(Rs_final),
        "cost": float(np.sum(res.fun ** 2)),
    }


def main():
    """Main execution function: runs the simplified fit and prints the result."""
    print("=" * 80)
    print("Starting Final Fit: Optimizing for a single global Rs only")
    print(f"Database: {DB_PATH}")
    print(f"Tables: {', '.join(TABLES)}")
    print(f"Fixed Rr value: {RR_FIXED:.4f} Ω")
    print("-" * 80)

    try:
        out = fit_only_rs(DB_PATH, TABLES)

        print("\nFinal Fitted Parameter:")
        print(f"  - Global Rs = {out['Rs']:.6e} Ω")

        print("\nBoundary Check:")
        warn_if_at_bounds(out['Rs'], RS_MIN, RS_MAX, "Global Rs")

        print(f"\nTotal Final Cost = {out['cost']:.4e}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        import sys
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Fit completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()