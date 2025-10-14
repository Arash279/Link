#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid joint fit with Rs prior regularization (Scheme B):
- A single, shared Rs is fitted across all experiments.
- A separate Rr is fitted for each experiment.
- Add a prior (regularization) on Rs: sqrt(lambda)*(Rs - RS_PRIOR)

Fixed parameters: Llr, Lm (paper-correct low-frequency T-equivalent).
Fitted vector: [Rs_shared, Rr_exp1, Rr_exp2, ...]

Original base: your previous hybrid fit. (This file is a drop-in replacement.)
"""

import os, sqlite3, math
from typing import Tuple, List, Dict
import numpy as np

# SciPy is required
try:
    from scipy.optimize import least_squares
except ImportError:
    raise RuntimeError("Please install SciPy: pip install scipy")

# ======== USER INPUTS / CONFIGURATION ========

DB_PATH = r"D:\Desktop\data\AP_1p5.db"
TABLES = ["exp_7", "exp_8", "exp_9"]

# Fixed motor constants (low-frequency model; s=1)
Lls   = 2.55e-2   # H  (full stator leakage in the series branch)
Llr   = 2.55e-2   # H  (rotor leakage inductance, fixed)
Lm    = 2.55e-1   # H  (magnetizing inductance, fixed)
Rcore = 4751.342  # Ohm (set INCLUDE_RCORE=False to ignore its branch)
INCLUDE_RCORE = True

# Fitting frequency window (well below fr=27.49 kHz)
F_FIT_MIN = 100.0
F_FIT_MAX = 500.0

# Parameter bounds for variables being fitted
RS_MIN, RS_MAX = 0.1, 20.0
RR_MIN, RR_MAX = 0.1, 80.0

# Initial guesses
RS0 = 2.0   # Ohm
RR0 = 5.0   # Ohm

# ---------- Rs PRIOR (Scheme B) ----------
# Use a prior / regularization on Rs to keep it near a plausible value.
USE_RS_PRIOR = True
RS_PRIOR     = 1.50   # Ω  (pick from experience, DC 4-wire, or extrapolation)
LAMBDA_RS    = 10.0   # weight of the prior; increase to pull Rs closer to RS_PRIOR
# -----------------------------------------

# --- Helper: boundary warnings ---
def warn_if_at_bounds(val, lo, hi, name, tol=1e-4):
    """Prints a warning if a value is near its lower/upper bound."""
    if abs(val - lo) / max(abs(lo), 1e-12) < tol:
        print(f"  [warn] {name} @ lower bound ({val:.3e} ≈ {lo:.3e})")
    if abs(val - hi) / max(abs(hi), 1e-12) < tol:
        print(f"  [warn] {name} @ upper bound ({val:.3e} ≈ {hi:.3e})")

# --------- Data loader ----------
def load_table(db_path: str, table: str, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
            rows = cur.fetchall()
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error in table '{table}': {e}")
    if not rows:
        raise ValueError(f"No rows in table '{table}'")
    arr = np.array(rows, dtype=float)
    f, zabs, ph_deg = arr[:, 0], arr[:, 1], arr[:, 2]
    mask = (f >= fmin) & (f <= fmax)
    f, zabs, ph_deg = f[mask], zabs[mask], ph_deg[mask]
    if len(f) == 0:
        raise ValueError(f"No samples in [{fmin},{fmax}] Hz for table '{table}'")
    Z = zabs * np.exp(1j * np.deg2rad(ph_deg))
    return f, Z

# --------- Low-frequency model ----------
def Z_branch(omega: np.ndarray, Rs: float, Lls: float, Rr: float, Llr: float,
             Lm: float, Rcore: float, include_Rcore: bool) -> np.ndarray:
    """
    Paper-correct low-frequency T-equivalent, s=1:
      Z = Rs + jωLls + [ 1/(jωLm) + (1/Rcore?) + 1/(Rr + jωLlr) ]^{-1}
    """
    jw = 1j * omega
    Zs = Rs + jw * Lls
    Zr = Rr + jw * Llr
    Y  = 1.0 / (jw * Lm) + 1.0 / Zr
    if include_Rcore and Rcore > 0:
        Y += 1.0 / Rcore
    return Zs + (1.0 / Y)

# --------- Hybrid joint fit (shared Rs, separate Rr[k]) ----------
def hybrid_joint_fit_with_rs_prior(db_path: str, tables: List[str]) -> Dict:
    data = []
    for t in tables:
        f, Z = load_table(db_path, t, F_FIT_MIN, F_FIT_MAX)
        w = 1.0 / np.maximum(f, 1.0)  # emphasize lower frequencies
        w /= np.mean(w)
        data.append({"table": t, "f": f, "w": w, "Z": Z})
    K = len(data)

    # Parameter vector: [Rs_shared, Rr_exp1, ..., Rr_expK]
    x0 = [RS0] + [RR0] * K
    lb = [RS_MIN] + [RR_MIN] * K
    ub = [RS_MAX] + [RR_MAX] * K

    def residual(x: np.ndarray) -> np.ndarray:
        Rs_shared = x[0]
        Rr_sep    = x[1:]
        res_list = []

        for k, rec in enumerate(data):
            omega = 2 * math.pi * rec["f"]
            Zm = rec["Z"]
            Zm_hat = Z_branch(omega,
                              Rs=Rs_shared, Lls=Lls, Rr=Rr_sep[k], Llr=Llr,
                              Lm=Lm, Rcore=Rcore, include_Rcore=INCLUDE_RCORE)
            dr = rec["w"] * (Zm_hat.real - Zm.real)
            di = rec["w"] * (Zm_hat.imag - Zm.imag)
            res_list.extend([dr, di])

        # ---- Rs prior regularization: sqrt(lambda)*(Rs - RS_PRIOR) ----
        if USE_RS_PRIOR:
            res_list.append(np.array([math.sqrt(LAMBDA_RS) * (Rs_shared - RS_PRIOR)], dtype=float))

        return np.concatenate(res_list)

    res = least_squares(residual, x0, bounds=(lb, ub), method="trf", loss="soft_l1")

    Rs_final = res.x[0]
    Rr_final = res.x[1:]

    return {
        "Rs": float(Rs_final),
        "Rr_list": [float(v) for v in Rr_final],
        "cost": float(np.sum(res.fun ** 2)),
        "tables": tables,
        "status": res.status,
        "nfev": res.nfev,
        "message": res.message,
    }

def main():
    print("=" * 80)
    print("Hybrid Joint Fit (Shared Rs + Rs Prior, Separate Rr)")
    print(f"DB: {DB_PATH}")
    print(f"Tables: {', '.join(TABLES)}")
    print(f"Freq Range: [{F_FIT_MIN:.0f}, {F_FIT_MAX:.0f}] Hz")
    print(f"Model: Low-frequency T-equivalent, s=1 | INCLUDE_RCORE={INCLUDE_RCORE}")
    print("-" * 80)
    if USE_RS_PRIOR:
        print(f"Rs prior: RS_PRIOR={RS_PRIOR:.3f} Ω, LAMBDA_RS={LAMBDA_RS:.2f}")
    else:
        print("Rs prior: disabled")
    print("-" * 80)

    try:
        out = hybrid_joint_fit_with_rs_prior(DB_PATH, TABLES)

        print("\nGlobal Shared Parameter:")
        print(f"  - Rs (fitted) = {out['Rs']:.6e} Ω")

        print("\nPer-Experiment Fitted Parameters:")
        for i, tbl in enumerate(out['tables']):
            print(f"  - [{tbl}] Rr = {out['Rr_list'][i]:.6e} Ω")

        print("\nBoundary Checks:")
        warn_if_at_bounds(out['Rs'], RS_MIN, RS_MAX, "Shared Rs")
        for i, tbl in enumerate(out['tables']):
            warn_if_at_bounds(out['Rr_list'][i], RR_MIN, RR_MAX, f"Rr for {tbl}")

        print(f"\nTotal Final Cost = {out['cost']:.4e}")
        print(f"Optimizer: status={out['status']}, nfev={out['nfev']}, msg='{out['message']}'")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        import sys
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Fit completed.")
    print("=" * 80)

if __name__ == "__main__":
    main()
