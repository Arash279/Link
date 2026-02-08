# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) Utilities (math helpers)
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

def fmt_Z(name: str, Z: complex) -> str:
    """Format complex impedance with magnitude and phase(deg)."""
    mag = abs(Z)
    ph = np.angle(Z, deg=True)
    ph = (ph + 180.0) % 360.0 - 180.0  # wrap to (-180,180]
    return f"{name}: {Z.real:+.6e}{Z.imag:+.6e}j |Z|={mag:.6e} ∠{ph:.2f}°"

def phase_diff_deg(phi_sim: np.ndarray, phi_exp: np.ndarray) -> np.ndarray:
    """
    Smallest signed difference between two phases (deg), result in (-180, 180].
    Avoids discontinuity issues in residual.
    """
    d = phi_sim - phi_exp
    return (d + 180.0) % 360.0 - 180.0


# ============================================================
# 1) Parameter vector (11D) + mapping
# ============================================================

PARAM_NAMES: List[str] = [
    "Lls", "Csw", "Rsw", "Llr", "Rrs", "Rcore",
    "Lm", "nLls", "Csf", "Rsf", "Csf0", "Lad"
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
    Lad: float   # 新增

    @staticmethod
    def from_vector(x: np.ndarray) -> "Params":
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != N_PARAMS:
            raise ValueError(f"Expected {N_PARAMS} params, got {x.size}")
        d = dict(zip(PARAM_NAMES, x.tolist()))
        return Params(**d)

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in PARAM_NAMES], dtype=float)

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in PARAM_NAMES}


# ============================================================
# 2) Impedance model definition (vectorized)
#    (structure consistent with your existing test.py)
# ============================================================

Rs = 8.703  # stator resistance (Ohm), added in series with Zmid parallel branch

def Zlad(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zlad = jωLad"""
    return 1j * omega * p.Lad

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
    # --- Y arms (same as before) ---
    Za = Z_nLls(omega, p)       # a
    Zb = Zmin(omega, p)         # b
    Zc = Zbra(omega, p)         # c

    # --- Y -> Δ (same as before) ---
    Z1, Z2, Z3 = Y_to_Delta(Za, Zb, Zc)

    Z_csf0_only = Zcsf0(omega, p)  # 只保留 Csf0
    Z4_0 = par3(Z3, 0.5 * Z3, Z_csf0_only)

    # ============================================================
    # UPDATED: redefine Δ edges and do ONE Δ -> Y
    #
    # Your mapping:
    #   Z2 : a-b edge
    #   Z1 : a-c edge
    #   Z4_0 : b-c edge
    #
    # delta_to_Y expects edges: (Zab, Zbc, Zca) = (a-b, b-c, c-a)
    # so we pass: Zab=Z2, Zbc=Z4_0, Zca=Z1
    # ============================================================
    Za1, Zb1, Zc1 = delta_to_Y(Zab=Z2, Zbc=Z4_0, Zca=Z1)

    # Star arms:
    Z4, Z5, Z6 = Za1, Zb1, Zc1

    # Keep return signature compatible (unused items set to None)
    Z7 = Z8 = Z9 = None
    return Z1, Z2, Z3, Z4_0, Z4, Z5, Z6, Z7, Z8, Z9


def Z_total(omega: np.ndarray, p: Params) -> np.ndarray:
    Z1, Z2, _, _, Z4, Z5, Z6, _, _, _ = Z1_to_Z9(omega, p)
    Z_parallel = par(Z6 + 0.5 * Z1, Z5 + 0.5 * Z2)

    Z_core_total = Z_parallel + Z4          # 原来的总阻抗（不含 Lad）
    return Zlad(omega, p) + Z_core_total + 0.5 * Zlad(omega, p)   # 首尾加1.5个 Lad（BC端口的两个短接）


# ============================================================
# 3) Load experiment data from SQLite
# ============================================================

def load_experiment_from_db(db_path: str, table: str = "exp_10") -> pd.DataFrame:
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
# 5) Fitting (least squares on log-magnitude + phase)
# ============================================================

def simulate_on_freq(f_hz: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (log10(|Z|), phase_deg_wrapped)
    """
    omega = 2.0 * np.pi * f_hz
    Z = Z_total(omega, p)
    mag = np.abs(Z)
    ph = np.angle(Z, deg=True)
    return np.log10(mag), wrap_phase_deg(ph)

def make_initial_params() -> Params:
    # your given initial values
    return Params(
        Lls=0.0261536,
        Csw = 8.68416e-10,
        Rsw = 14776.8,
        Llr = 0.0699247,
        Rrs = 283.057,
        Rcore = 4129.66,
        Lm = 0.0457641,
        nLls = 1.7806e-11,
        Csf = 3.12877e-10,
        Rsf = 27.4,
        Csf0 = 7.38e-09,
        Lad=1e-9,
    )

# ============================================================
# 6) Plotting (comparison)
# ============================================================

def plot_compare(
    f_exp: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp: np.ndarray,
    f_sim: np.ndarray,
    zabs_sim: np.ndarray,
    phase_sim: np.ndarray,
    title_suffix: str = "",
):
    plt.figure(figsize=(12, 8))

    # magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f_sim, np.log10(zabs_sim), label="Simulation", linewidth=2)
    plt.semilogx(f_exp, np.log10(zabs_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(f"Impedance Magnitude Comparison {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    # phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f_sim, wrap_phase_deg(phase_sim), label="Simulation", linewidth=2)
    plt.semilogx(f_exp, wrap_phase_deg(phase_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Impedance Phase Comparison {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Main
# ============================================================

def main():
    # ---- user config ----
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"

    # 画图时的仿真频点
    N_PLOT = 4000

    # ---- 这里改为：外部“传参”（你可以改成从文件/命令行读）
    # 方式1：直接传 dict（推荐，可读性最好）
    p_in = {
        "Lls": 0.0261536,
        "Csw": 8.68416e-10,
        "Rsw": 1.47768e4,
        "Llr": 0.0699247,
        "Rrs": 283.057,
        "Rcore": 4129.66,
        "Lm": 0.0457641,
        "nLls": 1.7806e-11,
        "Csf": 3.12877e-10,
        "Rsf": 27.4,
        "Csf0": 7.38e-09,
        "Lad": 1e-9,
    }
    p = Params(**p_in)

    # 如果你更想传 vector，也可以：
    # x = np.array([ ... 11个参数，顺序必须与 PARAM_NAMES 一致 ... ], dtype=float)
    # p = Params.from_vector(x)

    # ---- load experiment ----
    exp = load_experiment_from_db(DB_PATH, TABLE)
    f_all = exp["Freq"].to_numpy(dtype=float)
    zabs_all = exp["Zabs"].to_numpy(dtype=float)
    phase_all = exp["Phase"].to_numpy(dtype=float)

    # ---- simulate on dense grid ----
    f_plot = np.logspace(np.log10(f_all.min()), np.log10(f_all.max()), N_PLOT)
    logmag_sim, phase_sim = simulate_on_freq(f_plot, p)
    zabs_sim = 10 ** logmag_sim

    # ---- plot compare (simulation vs experiment) ----
    plot_compare(
        f_exp=f_all, zabs_exp=zabs_all, phase_exp=phase_all,
        f_sim=f_plot, zabs_sim=zabs_sim, phase_sim=phase_sim,
        title_suffix="(Given Params, no fitting)"
    )

    # ============================================================
    # Lad 扫描：以 1e-7 为中心，缩小到 ±0.5 dex (约 3.16e-8 -> 3.16e-7)，保持 9 个对数等距点
    # ============================================================
    Lad_list = np.logspace(np.log10(1e-7) - 0.5, np.log10(1e-7) + 0.5, 9)  # 扫描：约 3.16e-8 -> 3.16e-7（9 点）

    # 预计算实验数据（避免循环里重复算）
    logz_exp = np.log10(zabs_all)
    phase_exp_wrapped = wrap_phase_deg(phase_all)  # 你的程序里已有 wrap_phase_deg()

    # --- Figure 1: 幅值 (log10|Z|) 九宫格 ---
    fig_mag, axes_mag = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    axes_mag = axes_mag.ravel()

    # --- Figure 2: 相位 九宫格 ---
    fig_ph, axes_ph = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    axes_ph = axes_ph.ravel()

    for i, Lad in enumerate(Lad_list):
        Lad = float(Lad)
        p_run = Params(**{**p_in, "Lad": Lad})

        # 1) 每次画图前打印
        # ===== debug sample @ 2e7 Hz =====
        f0 = 2e7
        w0 = 2.0 * np.pi * f0
        omega0 = np.array([w0], dtype=float)

        # 基本支路
        Zlad0 = Zlad(omega0, p_run)[0]
        Zcsf00 = Zcsf0(omega0, p_run)[0]
        Z_csf0_ad0 = par(np.array([Zcsf00]), np.array([Zlad0]))[0]

        Z1_, Z2_, Z3_, Z4_0_arr, Z4_, Z5_arr, Z6_arr, *_ = Z1_to_Z9(omega0, p_run)

        Z1_ = Z1_[0]
        Z2_ = Z2_[0]
        Z4_00 = Z4_0_arr[0]
        Z4_ = Z4_[0]
        Z5_ = Z5_arr[0]
        Z6_ = Z6_arr[0]

        # 总阻抗
        Ztot0 = Z_total(omega0, p_run)[0]

        print(f"\n[Lad scan] i={i:02d}, Lad={Lad:.6e} H, sample @ f={f0:.3e} Hz")
        print("  " + fmt_Z("Zlad", Zlad0))
        print("  " + fmt_Z("Zcsf0", Zcsf00))
        print("  " + fmt_Z("Z4_0", Z4_00))
        print("  " + fmt_Z("Z_total", Ztot0))
        print("  " + fmt_Z("Z1", Z1_))
        print("  " + fmt_Z("Z2", Z2_))
        print("  " + fmt_Z("Z4 (series arm)", Z4_))
        print("  " + fmt_Z("Z5", Z5_))
        print("  " + fmt_Z("Z6", Z6_))

        logmag_sim, phase_sim = simulate_on_freq(f_plot, p_run)
        phase_sim_wrapped = wrap_phase_deg(phase_sim)

        # -------- 幅值小图 --------
        axm = axes_mag[i]
        axm.semilogx(f_plot, logmag_sim, label="Sim", linewidth=1.5)
        axm.semilogx(f_all, logz_exp, label="Exp", linewidth=1.0)
        axm.set_title(f"Lad={Lad:.6e} H")  # 显示更多小数位
        axm.grid(True)

        # -------- 相位小图 --------
        axp = axes_ph[i]
        axp.semilogx(f_plot, phase_sim_wrapped, label="Sim", linewidth=1.5)
        axp.semilogx(f_all, phase_exp_wrapped, label="Exp", linewidth=1.0)
        axp.set_title(f"Lad={Lad:.6e} H")  # 显示更多小数位
        axp.grid(True)

    # 只放一个 legend（避免每格都挤）
    axes_mag[0].legend(loc="best")
    axes_ph[0].legend(loc="best")

    # 全局标签与标题
    fig_mag.supxlabel("Frequency (Hz)")
    fig_mag.supylabel("log10(|Z|) (Ohm)")
    fig_mag.suptitle(f"Lad Scan ({Lad_list[0]:.2e} to {Lad_list[-1]:.2e}) - Magnitude", y=0.98)
    fig_mag.tight_layout()

    fig_ph.supxlabel("Frequency (Hz)")
    fig_ph.supylabel("Phase (deg, wrapped to -180~180)")
    fig_ph.suptitle(f"Lad Scan ({Lad_list[0]:.2e} to {Lad_list[-1]:.2e}) - Phase", y=0.98)
    fig_ph.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()