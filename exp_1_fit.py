# -*- coding: utf-8 -*-
"""
【中文说明】
本程序实现了一个用于感应电机高频等效电路的阻抗建模与参数拟合框架，
其中所有电路参数被统一视为一个 11 维可调参数向量。

程序整体采用模块化结构，主要分为以下几个部分：

(0) 工具函数（Utilities）：
    提供阻抗并联运算、相位包裹（phase wrapping）等基础数学工具，
    用于保证数值计算的稳定性与拟合过程中相位误差的连续性。

(1) 参数管理（Parameter handling）：
    使用 dataclass 定义 12 个物理参数及其向量表示，
    实现参数向量与具名物理量之间的双向映射，
    以提高可读性并方便数值优化。

(2) 阻抗模型定义（Impedance model definition）：
    以向量化形式实现高频等效电路的各个子阻抗，
    包括 Zmid、Zmr、Zbra 等基本模块，
    并通过 Y–Δ / Δ–Y 变换构建完整网络，
    最终得到总阻抗 Z_total(ω)。

(3) 实验数据读取（Experimental data loading）：
    从 SQLite 数据库中读取实验测得的阻抗幅值与相位数据，
    并进行必要的预处理（排序、去除无效点等）。

(4) 频率抽样（Frequency sampling）：
    在拟合阶段对实验频点进行对数均匀或随机抽样，
    以降低计算复杂度并加快参数优化过程。

(5) 参数拟合（Parameter fitting）：
    基于非线性最小二乘法，对模型阻抗与实验阻抗进行拟合，
    以 log(|Z|) 与相位为目标量，
    并通过对数域优化保证参数始终保持物理上的正值。

(6) 结果可视化（Visualization）：
    绘制拟合前后模型与实验阻抗在幅值与相位上的对比曲线，
    用于直观评估拟合效果。

(7) 主程序（Main routine）：
    统一调度数据读取、频率抽样、初始仿真、参数拟合及最终绘图流程。

------------------------------------------------------------

[English Description]
This script implements a high-frequency impedance modeling and parameter
fitting framework for an induction machine equivalent circuit, where all
circuit elements are treated as a single 11-dimensional tunable parameter vector.

The program is organized in a modular manner with the following main components:

(0) Utilities:
    Basic mathematical helper functions for impedance parallel operations
    and phase wrapping, ensuring numerical stability and phase continuity
    during optimization.

(1) Parameter handling:
    A dataclass-based definition of the 12 physical parameters,
    providing bidirectional mapping between a parameter vector and
    named circuit elements for readability and optimization convenience.

(2) Impedance model definition:
    Vectorized implementations of the high-frequency equivalent circuit,
    including elemental impedances (Zmid, Zmr, Zbra, etc.),
    Y–Δ / Δ–Y transformations, and the final total impedance Z_total(ω).

(3) Experimental data loading:
    Functions to load measured impedance magnitude and phase data
    from an SQLite database with basic preprocessing.

(4) Frequency sampling:
    Optional subsampling of experimental frequency points
    (e.g., log-uniform or random sampling) to reduce computational cost
    during parameter fitting.

(5) Parameter fitting:
    Nonlinear least-squares fitting of simulated impedance to experimental data
    using log-magnitude and phase as fitting targets,
    with positivity of parameters enforced via log-domain optimization.

(6) Visualization:
    Comparison plots of simulated and experimental impedance magnitude
    and phase before and after fitting.

(7) Main routine:
    Coordinates data loading, sampling, initial simulation,
    parameter optimization, and final visualization.
"""


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

def wrap_phase_deg(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    return (phi_deg + 180.0) % 360.0 - 180.0

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
# 2) Impedance model definition (SIMPLIFIED)
#    Use user-specified Ztotal
# ============================================================

Rs = 8.703  # keep as before

def Zmid(omega: np.ndarray, p: Params) -> np.ndarray:
    Z_L = 1j * omega * p.Lls
    Z_C = 1.0 / (1j * omega * p.Csw)
    Z_R = p.Rsw + 0j
    Z_par = 1.0 / (1.0 / Z_L + 1.0 / Z_C + 1.0 / Z_R)
    return Z_par + Rs

def Zmr(omega: np.ndarray, p: Params) -> np.ndarray:
    Z_series = 1j * omega * p.Llr + p.Rrs
    Z_core   = p.Rcore + 0j
    Z_Lm     = 1j * omega * p.Lm
    return 1.0 / (1.0/Z_series + 1.0/Z_core + 1.0/Z_Lm)

def Zmin(omega: np.ndarray, p: Params) -> np.ndarray:
    return Zmid(omega, p) + Zmr(omega, p)

def Z_nLls(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1j * omega * p.nLls

def Zbra(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1.0/(1j*omega*p.Csf) + p.Rsf

def Zcsf0(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1.0/(1j*omega*p.Csf0)

def Z_total(omega: np.ndarray, p: Params) -> np.ndarray:
    """
    Ztotal = ( ( (0.5*Zmin + 0.5*Zbra) || Zcsf0 ) + Zmin ) || Zbra + Z_nLls
    """
    Zm  = Zmin(omega, p)
    Zb  = Zbra(omega, p)
    Zc0 = Zcsf0(omega, p)
    Zn  = Z_nLls(omega, p)

    Z_a = 0.5 * Zm + 0.5 * Zb
    Z_b = par(Z_a, Zc0)
    Z_c = Z_b + Zm
    Z_d = par(Z_c, Zb)
    return Z_d + Zn

# ============================================================
# 3) Load experiment data from SQLite
# ============================================================

def load_experiment_from_db(db_path: str, table: str = "exp_1") -> pd.DataFrame:
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
# 4) Sampling strategy (for speed)
# ============================================================

def sample_freq_points(
    f_all: np.ndarray,
    zabs_all: np.ndarray,
    phase_all: np.ndarray,
    n_samples: int = 800,
    mode: str = "log_uniform",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick subset of experimental points to fit.
    mode:
      - "log_uniform": pick indices approximately uniform in log(f)
      - "random": random choice in index space
    """
    f_all = np.asarray(f_all)
    N = f_all.size
    if n_samples >= N:
        return f_all, zabs_all, phase_all

    if mode == "log_uniform":
        logf = np.log10(f_all)
        grid = np.linspace(logf.min(), logf.max(), n_samples)
        idx = np.searchsorted(logf, grid)
        idx = np.clip(idx, 0, N - 1)
        idx = np.unique(idx)
        # 如果 unique 之后点数少，再补一点
        if idx.size < n_samples:
            rng = np.random.default_rng(seed)
            extra = rng.choice(np.setdiff1d(np.arange(N), idx), size=min(n_samples-idx.size, N-idx.size), replace=False)
            idx = np.sort(np.concatenate([idx, extra]))
        return f_all[idx], zabs_all[idx], phase_all[idx]

    elif mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_samples, replace=False)
        idx = np.sort(idx)
        return f_all[idx], zabs_all[idx], phase_all[idx]

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


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

def default_bounds(p0: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple generic bounds to keep parameters physical & avoid divergence.
    You can tighten these later based on physics/experience.
    """
    x0 = p0.to_vector()

    # lower/upper as multipliers (for positive params)
    lo_mul = np.full(N_PARAMS, 0.1, dtype=float)
    hi_mul = np.full(N_PARAMS, 10.0, dtype=float)

    # resistances often vary wider
    for name in ["Rsw", "Rrs", "Rcore", "Rsf"]:
        i = PARAM_NAMES.index(name)
        lo_mul[i] = 0.01
        hi_mul[i] = 100.0

    lo = x0 * lo_mul
    hi = x0 * hi_mul

    # ensure strictly positive
    lo = np.maximum(lo, 1e-18)
    hi = np.maximum(hi, lo * 1.001)
    return lo, hi

def fit_params_least_squares(
    f_fit: np.ndarray,
    logmag_exp: np.ndarray,
    phase_exp: np.ndarray,
    p0: Params,
    w_logmag: float = 1.0,
    w_phase: float = 1.0 / 30.0,  # phase scaling: 30deg ~ 1 unit
    max_nfev: int = 200,
) -> Params:
    """
    Fit in log-parameter domain to enforce positivity:
      x = exp(u), optimize u
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as e:
        raise ImportError(
            "SciPy is required for fitting. Install it via: pip install scipy\n"
            f"Original error: {e}"
        )

    x0 = p0.to_vector()
    lo, hi = default_bounds(p0)

    # log-domain transform
    u0 = np.log(x0)
    ulo = np.log(lo)
    uhi = np.log(hi)

    logmag_exp = np.asarray(logmag_exp, float)
    phase_exp = wrap_phase_deg(np.asarray(phase_exp, float))

    def residual(u: np.ndarray) -> np.ndarray:
        x = np.exp(u)
        p = Params.from_vector(x)

        logmag_sim, phase_sim = simulate_on_freq(f_fit, p)
        r1 = (logmag_sim - logmag_exp) * w_logmag
        r2 = phase_diff_deg(phase_sim, phase_exp) * w_phase
        return np.concatenate([r1, r2])

    res = least_squares(
        residual,
        u0,
        bounds=(ulo, uhi),
        max_nfev=max_nfev,
        verbose=2,
    )

    p_opt = Params.from_vector(np.exp(res.x))
    return p_opt


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
    TABLE = "exp_1"          # 改成 exp_13 / exp_17 / exp_21 等也可以
    N_SAMPLES = 2000           # 抽样点数：越小越快，但可能更不稳
    SAMPLE_MODE = "log_uniform"  # "log_uniform" or "random"
    SEED = 0

    # 拟合时的权重
    W_LOGMAG = 1.0
    W_PHASE = 1.0 / 30.0      # 30deg -> 1 unit
    MAX_NFEV = 200

    # 画图时的仿真频点（用于展示，不一定等于拟合抽样点）
    N_PLOT = 4000

    # ---- load experiment ----
    exp = load_experiment_from_db(DB_PATH, TABLE)
    f_all = exp["Freq"].to_numpy(dtype=float)
    zabs_all = exp["Zabs"].to_numpy(dtype=float)
    phase_all = exp["Phase"].to_numpy(dtype=float)

    # ============================================================
    # 只用 10 ~ 1e7 Hz 参与拟合；忽略 1e7 ~ 1e8 Hz（以及 >1e7 的所有点）
    # ============================================================
    F_MIN = 10.0
    F_MAX = 1e7

    mask_fit = (f_all >= F_MIN) & (f_all <= F_MAX)

    f_fit_band = f_all[mask_fit]
    zabs_fit_band = zabs_all[mask_fit]
    phase_fit_band = phase_all[mask_fit]

    # ---- sample for fitting (only within 10~1e7) ----
    f_fit, zabs_fit, phase_fit = sample_freq_points(
        f_fit_band, zabs_fit_band, phase_fit_band,
        n_samples=N_SAMPLES,
        mode=SAMPLE_MODE,
        seed=SEED,
    )

    logmag_fit = np.log10(zabs_fit)

    # ---- initial model ----
    p0 = make_initial_params()

    # quick plot (initial vs exp)
    logmag0, phase0 = simulate_on_freq(f_fit, p0)
    # 初始对比图（用抽样点）
    plot_compare(
        f_exp=f_fit, zabs_exp=zabs_fit, phase_exp=phase_fit,
        f_sim=f_fit, zabs_sim=10**logmag0, phase_sim=phase0,
        title_suffix="(Initial, sampled points)"
    )

    # ---- fit ----
    p_opt = fit_params_least_squares(
        f_fit=f_fit,
        logmag_exp=logmag_fit,
        phase_exp=phase_fit,
        p0=p0,
        w_logmag=W_LOGMAG,
        w_phase=W_PHASE,
        max_nfev=MAX_NFEV,
    )

    print("\n===== Optimized parameters =====")
    for k, v in p_opt.as_dict().items():
        print(f"{k:8s} = {v:.6g}")

    # ---- plot final on a denser grid for readability ----
    # Use log-spaced points across experiment range
    f_plot = np.logspace(np.log10(f_all.min()), np.log10(f_all.max()), N_PLOT)
    logmag_sim, phase_sim = simulate_on_freq(f_plot, p_opt)
    zabs_sim = 10**logmag_sim

    plot_compare(
        f_exp=f_all, zabs_exp=zabs_all, phase_exp=phase_all,
        f_sim=f_plot, zabs_sim=zabs_sim, phase_sim=phase_sim,
        title_suffix="(Fitted)"
    )


if __name__ == "__main__":
    main()