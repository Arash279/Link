#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Induction Motor Model Parameter Identification
Strict replication of Mirafzal et al. (2007, 2009)
- Csw uses only CsfHF (Eq. 3 or 4)
- CsfLF used solely for Csf0 computation
- No heuristic or averaged modification
"""

import math
from typing import Literal

# ======================
# ===== USER INPUT =====
# ======================

CsfHF = 2.461e-10       # F, high-frequency effective stator-to-frame capacitance (one-lead-to-frame)
CsfLF = 1.516e-9        # F, low-frequency measured stator-to-frame capacitance (one-lead-to-frame)
Csf0_reported = 7.38e-10  # F, reported neutral-to-frame capacitance for comparison

Lls = 2.55e-2           # H, stator leakage inductance
Llr = 2.55e-2           # H, rotor leakage inductance (referred to stator)
Lm  = 5.5e-2           # H, magnetizing inductance (for completeness, not used in HF calc)
Rcore = 4751.342        # Ω, core-loss resistance from empirical hp relation

fr_hz = 36311.2         # Hz, first resonance frequency #exp_1
Zmax  = 2.50e4          # Ω, impedance magnitude at resonance peak
fa_hz = 66734.6         # Hz, antiresonance frequency #exp_1
Zanti = 4.11e3          # Ω, impedance magnitude at antiresonance

connection: Literal["Y", "Delta"] = "Y"  # winding connection type

# ======================
# ===== FORMULAS =======
# ======================

def csw_from_fr_Y(Csf: float, Lls: float, fr_hz: float) -> float:
    """ Eq.(3) from Mirafzal 2009, Y connection """
    w = 2 * math.pi * fr_hz
    num = 2 * (w**2) * Lls * Csf - 1.0
    den = (w**2) * Lls * ((w**2) * Lls * Csf - 1.0)
    return num / den

def csw_from_fr_Delta(Csf: float, Lls: float, fr_hz: float) -> float:
    """ Eq.(4) from Mirafzal 2009, Delta connection """
    w2 = 2 * (2 * math.pi * fr_hz)  # 2 * ωr
    num = 2 * (w2**2) * Lls * Csf - 1.0
    den = (w2**2) * Lls * ((w2**2) * Lls * Csf - 1.0)
    return num / den

def eta_Lls_from_fa(Csf_effective: float, fa_hz: float) -> float:
    """ Eq.(6) ηLls = 1 / [Csf (2π fa)^2] """
    return 1.0 / (Csf_effective * (2 * math.pi * fa_hz) ** 2)

def Rsf_from_Zanti(Zanti: float) -> float:
    """ Eq.(5) Rsf = (2/3) |Z|_anti-r """
    return (2.0 / 3.0) * Zanti

def parallel_mag_R_and_jX(R: float, X: float) -> float:
    """ magnitude of parallel of R and jX """
    return (R * abs(X)) / math.sqrt(R**2 + X**2)

def Rsw_from_Zmax_Rcore_Llr(fr_hz: float, Zmax: float, Rcore: float, Llr: float) -> float:
    """ Eq.(7) Rsw = (2/3)|Z|max - |Rcore || jωLlr| """
    X_L = 2 * math.pi * fr_hz * Llr
    Z_par = parallel_mag_R_and_jX(Rcore, X_L)
    return (2.0 / 3.0) * Zmax - Z_par

def Csf0_from_LF_HF(CsfLF: float, CsfHF: float) -> float:
    """ Eq.(8) Csf0 = Csf(LF) - 3*Csf(HF) """
    return CsfLF - 3.0 * CsfHF

# ======================
# ===== COMPUTATION ====
# ======================

def main():
    if connection == "Y":
        Csw = csw_from_fr_Y(CsfHF, Lls, fr_hz)
    elif connection == "Delta":
        Csw = csw_from_fr_Delta(CsfHF, Lls, fr_hz)
    else:
        raise ValueError("connection must be 'Y' or 'Delta'")

    eta_Lls = eta_Lls_from_fa(CsfHF, fa_hz)
    Rsf = Rsf_from_Zanti(Zanti)
    Rsw = Rsw_from_Zmax_Rcore_Llr(fr_hz, Zmax, Rcore, Llr)
    Csf0_calc = Csf0_from_LF_HF(CsfLF, CsfHF)

    def fmt(v: float, unit: str) -> str:
        return f"{v:.6e} {unit}"

    print("=== Universal Induction Motor Parameters (Paper Exact) ===\n")
    print("Inputs:")
    print(f"  Connection        : {connection}")
    print(f"  CsfHF             : {fmt(CsfHF, 'F')}")
    print(f"  CsfLF             : {fmt(CsfLF, 'F')}")
    print(f"  fr                : {fr_hz:.3f} Hz")
    print(f"  fa                : {fa_hz:.3f} Hz")
    print(f"  |Z|max            : {fmt(Zmax, 'Ω')}")
    print(f"  |Z|_anti-r        : {fmt(Zanti, 'Ω')}")
    print(f"  Lls               : {fmt(Lls, 'H')}")
    print(f"  Llr               : {fmt(Llr, 'H')}")
    print(f"  Rcore             : {fmt(Rcore, 'Ω')}\n")

    print("Results (exact per 2009 paper):")
    print(f"  Csw               : {fmt(Csw, 'F')}")
    print(f"  ηLls              : {fmt(eta_Lls, 'H')}")
    print(f"  Rsf               : {fmt(Rsf, 'Ω')}")
    print(f"  Rsw               : {fmt(Rsw, 'Ω')}")
    print(f"  Csf0 (calc)       : {fmt(Csf0_calc, 'F')} (reported {fmt(Csf0_reported, 'F')})\n")

    # Basic validation notes (not altering computation)
    if Csw <= 0:
        print("NOTE: Csw <= 0, check fr/Lls/CsfHF (term ω²LlsCsfHF - 1).")
    if eta_Lls > Lls:
        print("NOTE: ηLls > Lls; physically it should be a small fraction of Lls.")

if __name__ == "__main__":
    main()
