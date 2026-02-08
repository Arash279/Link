# -*- coding: utf-8 -*-
"""
compare_no_fit.py

【功能】
- 取消拟合环节：不再做 least_squares / 参数优化
- 直接使用一套“固定电路参数”进行仿真
- 从 SQLite DB 读取实验数据（Freq, Zabs, Phase）
- 输出对比图（log10|Z| 与 phase）与误差指标（RMSE / MAE 等）

【电路结构】
保持与你当前 exp_10_fit.py 一致的阻抗模型：
Z_total = (Z6 + 1/2 Z1) || (Z5 + 1/2 Z2) + Z4
以及 Zmid/Zmr/Zmin/Zbra/Zcsf0、Y-Δ / Δ-Y 变换等定义一致。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Utilities
# ============================================================

def par(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel of two impedances (vectorized)."""
    return a * b / (a + b)

def par3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Parallel of three impedances (vectorized)."""
    return 1.0 / (1.0/a + 1.0/b + 1.0/c)

def wrap_phase_deg(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    return (phi_deg + 180.0) % 360.0 - 180.0

def phase_diff_deg(phi_sim: np.ndarray, phi_exp: np.ndarray) -> np.ndarray:
    """
    Smallest signed difference between two phases (deg), result in (-180, 180].
    """
    d = phi_sim - phi_exp
    return (d + 180.0) % 360.0 - 180.0


# ============================================================
# 1) Parameter handling
# ============================================================

PARAM_NAMES: List[str] = [
    "Lls", "Csw", "Rsw", "Llr", "Rrs", "Rcore",
    "Lm", "nLls", "Csf", "Rsf", "Csf0"
]
N_PARAMS: int = len(PARAM_NAMES)

@dataclass
class Params:
    Lls: float
    Csw: float
    Rsw: float
    Llr: float
    Rrs: float
    Rcore: float
    Lm: float
    nLls: float
    Csf: float
    Rsf: float
    Csf0: float

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Params":
        missing = [k for k in PARAM_NAMES if k not in d]
        extra = [k for k in d.keys() if k not in PARAM_NAMES]
        if missing:
            raise ValueError(f"Missing params: {missing}")
        if extra:
            raise ValueError(f"Unknown params: {extra}")
        return Params(**{k: float(d[k]) for k in PARAM_NAMES})

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in PARAM_NAMES}


# ============================================================
# 2) Impedance model definition (same structure as exp_10_fit.py)
# ============================================================

Rs = 8.703  # stator resistance (Ohm), added in series with Zmid parallel branch

def Zmid(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmid = (jωLls) || (1/jωCsw) || Rsw  + Rs"""
    Z_L = 1j * omega * p.Lls
    Z_C = 1.0 / (1j * omega * p.Csw)
    Z_R = p.Rsw + 0j
    Z_par = 1.0 / (1.0 / Z_L + 1.0 / Z_C + 1.0 / Z_R)
    return Z_par + Rs

def Zmr(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmr = (jωLlr + Rrs) || Rcore || (jωLm)"""
    Z_series = 1j * omega * p.Llr + p.Rrs
    Z_core   = p.Rcore + 0j
    Z_Lm     = 1j * omega * p.Lm
    return 1.0 / (1.0/Z_series + 1.0/Z_core + 1.0/Z_Lm)

def Zmin(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmin = Zmid + Zmr"""
    return Zmid(omega, p) + Zmr(omega, p)

def Z_nLls(omega: np.ndarray, p: Params) -> np.ndarray:
    """Z_nLls = jω nLls"""
    return 1j * omega * p.nLls

def Zbra(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zbra = 1/(jωCsf) + Rsf  (Csf series Rsf)"""
    return 1.0/(1j*omega*p.Csf) + p.Rsf

def Zcsf0(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zcsf0 = 1/(jωCsf0)"""
    return 1.0/(1j*omega*p.Csf0)

def Y_to_Delta(Za: np.ndarray, Zb: np.ndarray, Zc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y -> Δ, vectorized.
    Returns edges Z1(a-b), Z2(a-c), Z3(b-c)
    """
    S = Za*Zb + Zb*Zc + Zc*Za
    Z1 = S / Zc
    Z2 = S / Zb
    Z3 = S / Za
    return Z1, Z2, Z3

def delta_to_Y(Zab: np.ndarray, Zbc: np.ndarray, Zca: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Δ -> Y, vectorized.
    Returns star arms Za(node a), Zb(node b), Zc(node c)
    """
    S = Zab + Zbc + Zca
    Za = Zab * Zca / S
    Zb = Zab * Zbc / S
    Zc = Zbc * Zca / S
    return Za, Zb, Zc

def Z1_to_Z9(omega: np.ndarray, p: Params):
    # Y arms
    Za = Z_nLls(omega, p)       # a
    Zb = Zmin(omega, p)         # b
    Zc = Zbra(omega, p)         # c

    # Y -> Δ
    Z1, Z2, Z3 = Y_to_Delta(Za, Zb, Zc)

    # Z4_0 = Z3 || (1/2 Z3) || Zcsf0
    Z4_0 = par3(Z3, 0.5 * Z3, Zcsf0(omega, p))

    # Δ edges mapping:
    #   Z2 : a-b edge
    #   Z1 : a-c edge
    #   Z4_0 : b-c edge
    # delta_to_Y expects (Zab, Zbc, Zca) = (a-b, b-c, c-a)
    Za1, Zb1, Zc1 = delta_to_Y(Zab=Z2, Zbc=Z4_0, Zca=Z1)

    # Star arms:
    Z4, Z5, Z6 = Za1, Zb1, Zc1

    Z7 = Z8 = Z9 = None
    return Z1, Z2, Z3, Z4_0, Z4, Z5, Z6, Z7, Z8, Z9

def Z_total(omega: np.ndarray, p: Params) -> np.ndarray:
    """
    Z_total = (Z6 + 1/2 Z1) || (Z5 + 1/2 Z2) + Z4
    """
    Z1, Z2, _, _, Z4, Z5, Z6, _, _, _ = Z1_to_Z9(omega, p)
    Z_parallel = par(Z6 + 0.5 * Z1, Z5 + 0.5 * Z2)
    return Z_parallel + Z4


# ============================================================
# 3) Load experiment data from SQLite
# ============================================================

def load_experiment_from_db(db_path: str, table: str) -> pd.DataFrame:
    """
    Expect columns: Freq, Zabs, Phase (deg)
    """
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT Freq, Zabs, Phase FROM {table}"
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    df = df.dropna().copy()
    df = df.sort_values("Freq")
    df = df[df["Freq"] > 0]
    return df


# ============================================================
# 4) Simulation + metrics
# ============================================================

def simulate_on_freq(f_hz: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Z (complex), log10(|Z|), phase_deg_wrapped
    """
    f_hz = np.asarray(f_hz, dtype=float)
    omega = 2.0 * np.pi * f_hz
    Z = Z_total(omega, p)
    mag = np.abs(Z)
    ph = np.angle(Z, deg=True)
    return Z, np.log10(mag), wrap_phase_deg(ph)

def compute_metrics(
    f: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp: np.ndarray,
    logmag_sim: np.ndarray,
    phase_sim: np.ndarray,
    f_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Error metrics on:
      - log10|Z|
      - phase (wrapped diff)
    If f_range given, compute on that band only.
    """
    f = np.asarray(f, float)
    zabs_exp = np.asarray(zabs_exp, float)
    phase_exp = wrap_phase_deg(np.asarray(phase_exp, float))

    if f_range is not None:
        f_lo, f_hi = f_range
        m = (f >= f_lo) & (f <= f_hi)
    else:
        m = np.ones_like(f, dtype=bool)

    if m.sum() < 5:
        raise ValueError("Too few points in selected frequency range.")

    logmag_exp = np.log10(zabs_exp[m])
    logmag_err = logmag_sim[m] - logmag_exp
    phase_err = phase_diff_deg(phase_sim[m], phase_exp[m])

    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    def mae(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))

    return {
        "N_points": int(m.sum()),
        "logmag_RMSE": rmse(logmag_err),
        "logmag_MAE": mae(logmag_err),
        "phase_RMSE_deg": rmse(phase_err),
        "phase_MAE_deg": mae(phase_err),
    }


# ============================================================
# 5) Plotting
# ============================================================

def plot_compare(
    f: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp: np.ndarray,
    zabs_sim: np.ndarray,
    phase_sim: np.ndarray,
    title: str,
):
    plt.figure(figsize=(12, 8))

    # magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f, np.log10(zabs_sim), label="Simulation", linewidth=2)
    plt.semilogx(f, np.log10(zabs_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(title + " - Magnitude")
    plt.grid(True)
    plt.legend()

    # phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f, wrap_phase_deg(phase_sim), label="Simulation", linewidth=2)
    plt.semilogx(f, wrap_phase_deg(phase_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(title + " - Phase")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Main (NO FIT)
# ============================================================

def main():
    # ----------------------------
    # User config
    # ----------------------------
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"   # 例如: exp_1 / exp_10 / exp_13 / exp_17 / exp_21 ...

    # 你要“固定使用”的那套参数：直接在这里填！
    # （下面这一套是你 exp_10_fit.py 里 make_initial_params 的数值，仅作占位示例）
    PARAMS_FIXED: Dict[str, float] = dict(
        Lls=2.55e-2,
        Csw=1.012e-9,
        Rsw=1.3437e4,
        Llr=2.55e-2,
        Rrs=28.0,
        Rcore=4.751e3,
        Lm=5.5e-2,
        nLls=1.7806e-10,
        Csf=2.461e-10,
        Rsf=2.74e3,
        Csf0=7.38e-10,
    )

    # 误差统计频段：None 表示全频
    METRIC_BANDS = [
        None,                    # 全频
        (10.0, 1e7),              # 10 ~ 10 MHz
        (1e7, 1e8),               # 10 MHz ~ 100 MHz（你关心的那段）
    ]

    # ----------------------------
    # Load experiment
    # ----------------------------
    exp = load_experiment_from_db(DB_PATH, TABLE)
    f = exp["Freq"].to_numpy(dtype=float)
    zabs_exp = exp["Zabs"].to_numpy(dtype=float)
    phase_exp = exp["Phase"].to_numpy(dtype=float)

    # ----------------------------
    # Simulate with fixed params
    # ----------------------------
    p = Params.from_dict(PARAMS_FIXED)

    Z, logmag_sim, phase_sim = simulate_on_freq(f, p)
    zabs_sim = np.abs(Z)

    # ----------------------------
    # Print params & metrics
    # ----------------------------
    print("\n===== Fixed parameters used (NO FIT) =====")
    for k, v in p.as_dict().items():
        print(f"{k:8s} = {v:.6g}")

    print("\n===== Error metrics =====")
    for band in METRIC_BANDS:
        if band is None:
            tag = "Full band"
        else:
            tag = f"{band[0]:.3g} ~ {band[1]:.3g} Hz"
        m = compute_metrics(
            f=f,
            zabs_exp=zabs_exp,
            phase_exp=phase_exp,
            logmag_sim=logmag_sim,
            phase_sim=phase_sim,
            f_range=band,
        )
        print(f"\n[{tag}]  N={m['N_points']}")
        print(f"  log10|Z| RMSE = {m['logmag_RMSE']:.6g}")
        print(f"  log10|Z| MAE  = {m['logmag_MAE']:.6g}")
        print(f"  Phase RMSE(deg)= {m['phase_RMSE_deg']:.6g}")
        print(f"  Phase MAE(deg) = {m['phase_MAE_deg']:.6g}")

    # ----------------------------
    # Plot
    # ----------------------------
    plot_compare(
        f=f,
        zabs_exp=zabs_exp,
        phase_exp=phase_exp,
        zabs_sim=zabs_sim,
        phase_sim=phase_sim,
        title=f"Impedance Compare (NO FIT) - {TABLE}"
    )


if __name__ == "__main__":
    main()
