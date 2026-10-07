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
    使用 dataclass 定义 11 个物理参数及其向量表示，
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
    A dataclass-based definition of the 11 physical parameters,
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
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class RunStats:
    # counts
    model_eval: int = 0
    residual_calls: int = 0
    objective_calls: int = 0
    # timing (seconds)
    t_global: float = 0.0
    t_local: float = 0.0
    t_total_fit: float = 0.0
    t_gp: float = 0.0


STATS = RunStats()


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

def phase_diff_deg(phi_sim: np.ndarray, phi_exp: np.ndarray) -> np.ndarray:
    """
    Smallest signed difference between two phases (deg), result in (-180, 180].
    Avoids discontinuity issues in residual.
    """
    d = phi_sim - phi_exp
    return (d + 180.0) % 360.0 - 180.0

def mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled for robustness."""
    x = np.asarray(x, float)
    med = np.median(x)
    return np.median(np.abs(x - med))

def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Convert magnitude + phase(deg) to complex."""
    return mag * np.exp(1j * np.deg2rad(phase_deg))


# ============================================================
# 1) Parameter vector (11D) + mapping
# ============================================================

PARAM_NAMES: List[str] = [
    "Lls", "Csw", "Rsw", "Llr", "Rrs", "Rcore",
    "Lm", "nLls", "Csf", "Rsf", "Csf0", "Lad"   # 添加 Lad
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
    Lad: float   # 新增 Lad

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
#    (structure consistent with your existing Cs.py)
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

def Zlad(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zlad = jωLad"""
    return 1j * omega * p.Lad

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

    # --- Z4_0 = Z3 || (1/2 Z3) || Zcsf0 (same as before) ---
    Z4_0 = par3(Z3, 0.5 * Z3, Zcsf0(omega, p))

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
    """
    UPDATED:
    Z_total = Lad (series) + [ (Z6 + 1/2 Z1) || (Z5 + 1/2 Z2) + Z4 ] + Lad (series)
    """
    Z1, Z2, _, _, Z4, Z5, Z6, _, _, _ = Z1_to_Z9(omega, p)
    Z_parallel = par(Z6 + 0.5 * Z1, Z5 + 0.5 * Z2)
    Z_core_total = Z_parallel + Z4
    return Zlad(omega, p) + Z_core_total + 0.5 * Zlad(omega, p)  # 首尾加1.5个 Lad（BC端口的两个短接）

# ============================================================
# 3) Load experiment data from SQLite
# ============================================================

def load_experiment_from_db(db_path: str, table: str = "exp_10", max_freq: float = 1e8) -> pd.DataFrame:
    """
    Expect columns: Freq, Zabs, Phase (deg)
    默认只读取频率区间 (0, max_freq]，max_freq 默认 1e8 Hz
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
    df = df[df["Freq"] <= max_freq]  # 限制上限到 1e8 Hz（默认）
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
# 5) Fitting (global -> local, Re/Im residual, robust loss)
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

def simulate_complex(f_hz: np.ndarray, p: Params) -> np.ndarray:
    """Return complex impedance Z on frequencies."""
    STATS.model_eval += 1
    omega = 2.0 * np.pi * f_hz
    return Z_total(omega, p)

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
        Lad=1.3e-7,
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

def compute_freq_weights(
    f_hz: np.ndarray,
    Z_data: np.ndarray,
    mode: str = "auto",
    min_w: float = 0.2,
    max_w: float = 5.0,
    power: float = 1.0,
) -> np.ndarray:
    """
    Frequency weights for residuals.
    mode:
      - "auto": emphasize regions with large slope in log|Z| or phase
      - "none": uniform weights
    """
    if mode == "none":
        return np.ones_like(f_hz, dtype=float)
    if mode != "auto":
        raise ValueError(f"Unknown weight mode: {mode}")

    f_hz = np.asarray(f_hz, float)
    logf = np.log10(f_hz)
    logmag = np.log10(np.abs(Z_data))
    phase_rad = np.unwrap(np.angle(Z_data))

    dlogmag = np.gradient(logmag, logf)
    dphase = np.gradient(phase_rad, logf)
    score = np.abs(dlogmag) + np.abs(dphase)
    score_med = np.median(score) + 1e-12
    w = score / score_med
    w = np.clip(w, min_w, max_w)
    if power != 1.0:
        w = w ** power
    return w

def make_residual_fn(
    f_hz: np.ndarray,
    Z_data: np.ndarray,
    weights: np.ndarray,
    s_re: float,
    s_im: float,
):
    f_hz = np.asarray(f_hz, float)
    Z_data = np.asarray(Z_data, complex)
    weights = np.asarray(weights, float)

    def residual(u: np.ndarray) -> np.ndarray:
        STATS.residual_calls += 1
        x = np.exp(u)
        p = Params.from_vector(x)
        Z_sim = simulate_complex(f_hz, p)
        r_re = weights * (Z_sim.real - Z_data.real) / s_re
        r_im = weights * (Z_sim.imag - Z_data.imag) / s_im
        return np.concatenate([r_re, r_im])

    return residual

def sample_initials(
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u_lo = np.asarray(u_lo, float)
    u_hi = np.asarray(u_hi, float)
    return rng.uniform(u_lo, u_hi, size=(n_samples, u_lo.size))

def fit_params_global_local(
    f_fit: np.ndarray,
    Z_fit: np.ndarray,
    p0: Params,
    weights: np.ndarray,
    s_re: float,
    s_im: float,
    n_starts: int = 100,
    top_k: int = 10,
    seed: int = 0,
    global_method: str = "de",
    max_nfev: int = 200,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    de_maxiter: int = 60,
    de_popsize: int = 10,
) -> Tuple[Params, List[dict]]:
    """
    Global -> local refinement in log-domain.
    Returns best params and a list of result dicts.
    """
    try:
        from scipy.optimize import least_squares, differential_evolution
    except ImportError as e:
        raise ImportError(
            "SciPy is required for fitting. Install it via: pip install scipy\n"
            f"Original error: {e}"
        )

    x0 = p0.to_vector()
    lo, hi = default_bounds(p0)
    u0 = np.log(x0)
    u_lo = np.log(lo)
    u_hi = np.log(hi)

    residual = make_residual_fn(f_fit, Z_fit, weights, s_re, s_im)

    def objective(u: np.ndarray) -> float:
        STATS.objective_calls += 1
        r = residual(u)
        return 0.5 * float(np.dot(r, r))

    candidates = []

    if global_method == "de":
        bounds = list(zip(u_lo.tolist(), u_hi.tolist()))
        t = time.perf_counter()
        de_res = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            polish=False,
            seed=seed,
        )
        STATS.t_global += time.perf_counter() - t
        candidates.append(de_res.x)
    else:
        raise ValueError(f"Unknown global method: {global_method}")

    u_samples = sample_initials(u_lo, u_hi, n_starts, seed=seed + 1)
    scores = np.array([objective(u) for u in u_samples], dtype=float)
    top_idx = np.argsort(scores)[:max(top_k, 1)]
    candidates.extend([u_samples[i] for i in top_idx])

    results = []
    for u_start in candidates:
        t = time.perf_counter()
        res = least_squares(
            residual,
            u_start,
            bounds=(u_lo, u_hi),
            max_nfev=max_nfev,
            loss=loss,
            f_scale=f_scale,
            verbose=0,
        )
        STATS.t_local += time.perf_counter() - t
        results.append(
            {
                "x": res.x,
                "cost": res.cost,
                "status": res.status,
                "message": res.message,
                "nfev": res.nfev,
            }
        )

    results.sort(key=lambda d: d["cost"])
    best = results[0]
    p_best = Params.from_vector(np.exp(best["x"]))
    return p_best, results

def compute_aic_bic(rss: float, n: int, p: int) -> Tuple[float, float]:
    rss = max(rss, 1e-24)
    aic = n * np.log(rss / n) + 2 * p
    bic = n * np.log(rss / n) + p * np.log(n)
    return aic, bic

def block_split_indices(n: int, n_blocks: int, n_val_blocks: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into contiguous blocks; select some blocks for validation.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    blocks = np.array_split(idx, n_blocks)
    val_blocks = rng.choice(len(blocks), size=n_val_blocks, replace=False)
    val_idx = np.concatenate([blocks[i] for i in val_blocks])
    train_idx = np.setdiff1d(idx, val_idx, assume_unique=True)
    return train_idx, val_idx

def gp_residual_analysis(
    f_hz: np.ndarray,
    Z_data: np.ndarray,
    p_opt: Params,
    weights: Optional[np.ndarray] = None,
    out_prefix: str = "gp_residual",
    top_n: int = 3,
):
    t0 = time.perf_counter()
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        print("scikit-learn not available; skip GP residual analysis.")
        return

    Z_sim = simulate_complex(f_hz, p_opt)
    res_re = (Z_data.real - Z_sim.real)
    res_im = (Z_data.imag - Z_sim.imag)

    logf = np.log10(f_hz)
    x = (logf - logf.mean()) / (logf.std() + 1e-12)
    x = x.reshape(-1, 1)

    def robust_zscore(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        med = np.median(v)
        scale = mad(v)
        scale = max(scale, 1e-12)
        return 0.6745 * (v - med) / scale

    z_re = np.abs(robust_zscore(res_re))
    z_im = np.abs(robust_zscore(res_im))
    keep = (z_re <= 5.0) & (z_im <= 5.0)
    if np.sum(keep) < 10:
        keep = np.ones_like(z_re, dtype=bool)

    x_gp = x[keep]
    res_re_gp = res_re[keep]
    res_im_gp = res_im[keep]

    kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6)
    gp_re = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp_im = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

    gp_re.fit(x_gp, res_re_gp)
    gp_im.fit(x_gp, res_im_gp)

    logf_grid = np.linspace(logf.min(), logf.max(), 400)
    x_grid = (logf_grid - logf.mean()) / (logf.std() + 1e-12)
    f_grid = 10 ** logf_grid
    mean_re, std_re = gp_re.predict(x_grid.reshape(-1, 1), return_std=True)
    mean_im, std_im = gp_im.predict(x_grid.reshape(-1, 1), return_std=True)

    def top_freqs(mean_vec: np.ndarray) -> List[float]:
        order = np.argsort(np.abs(mean_vec))[::-1]
        picked = []
        for idx in order:
            f0 = f_grid[idx]
            if all(abs(np.log10(f0 / fp)) > 0.15 for fp in picked):
                picked.append(f0)
            if len(picked) >= top_n:
                break
        return picked

    top_re = top_freqs(mean_re)
    top_im = top_freqs(mean_im)

    print("GP structure peaks (Re):", ", ".join(f"{f0:.3g} Hz" for f0 in top_re))
    print("GP structure peaks (Im):", ", ".join(f"{f0:.3g} Hz" for f0 in top_im))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].semilogx(f_hz, res_re, ".", alpha=0.4, label="Residual Re")
    axes[0].semilogx(f_grid, mean_re, "r-", label="GP mean")
    axes[0].fill_between(f_grid, mean_re - std_re, mean_re + std_re, color="r", alpha=0.2)
    axes[0].set_ylabel("Re residual (Ohm)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].semilogx(f_hz, res_im, ".", alpha=0.4, label="Residual Im")
    axes[1].semilogx(f_grid, mean_im, "r-", label="GP mean")
    axes[1].fill_between(f_grid, mean_im - std_im, mean_im + std_im, color="r", alpha=0.2)
    axes[1].set_ylabel("Im residual (Ohm)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.show()
    STATS.t_gp += time.perf_counter() - t0


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
    TABLE = "exp_10"          # ??? exp_13 / exp_17 / exp_21 ??????
    N_SAMPLES = 2000           # ????????????????????????
    SAMPLE_MODE = "log_uniform"  # "log_uniform" or "random"
    SEED = 0

    # multi-start + global -> local
    N_STARTS = 120
    TOP_K = 10
    GLOBAL_METHOD = "de"
    MAX_NFEV = 200
    DE_MAXITER = 60
    DE_POPSIZE = 10

    # residual settings (Re/Im + frequency weights)
    WEIGHT_MODE = "auto"   # "auto" or "none"
    WEIGHT_MIN = 0.3
    WEIGHT_MAX = 4.0
    WEIGHT_POWER = 1.0
    SCALE_MODE = "mad"     # "mad" or "std"

    # robust loss
    LOSS = "soft_l1"       # "soft_l1" or "huber"
    F_SCALE = None         # None -> estimate from initial residuals

    # ??????????????????????????????????????
    N_PLOT = 4000
    DO_VAL = False
    VAL_BLOCKS = 6
    VAL_HOLDOUT = 1

    # ---- load experiment ----
    exp = load_experiment_from_db(DB_PATH, TABLE)
    f_all = exp["Freq"].to_numpy(dtype=float)
    zabs_all = exp["Zabs"].to_numpy(dtype=float)
    phase_all = exp["Phase"].to_numpy(dtype=float)
    Z_all = mag_phase_to_complex(zabs_all, phase_all)

    # ---- sample for fitting ----
    f_fit, zabs_fit, phase_fit = sample_freq_points(
        f_all, zabs_all, phase_all,
        n_samples=N_SAMPLES,
        mode=SAMPLE_MODE,
        seed=SEED,
    )

    Z_fit = mag_phase_to_complex(zabs_fit, phase_fit)

    # ---- initial model ----
    p0 = make_initial_params()

    # quick plot (initial vs exp)
    logmag0, phase0 = simulate_on_freq(f_fit, p0)
    # ?????????????????
    plot_compare(
        f_exp=f_fit, zabs_exp=zabs_fit, phase_exp=phase_fit,
        f_sim=f_fit, zabs_sim=10**logmag0, phase_sim=phase0,
        title_suffix="(Initial, sampled points)"
    )

    # ---- fit (global -> local) ----
    m0 = STATS.model_eval
    r0 = STATS.residual_calls
    o0 = STATS.objective_calls
    t_fit0 = time.perf_counter()
    weights_fit = compute_freq_weights(
        f_fit, Z_fit, mode=WEIGHT_MODE, min_w=WEIGHT_MIN, max_w=WEIGHT_MAX, power=WEIGHT_POWER
    )
    if SCALE_MODE == "mad":
        s_re = mad(Z_fit.real)
        s_im = mad(Z_fit.imag)
    elif SCALE_MODE == "std":
        s_re = float(np.std(Z_fit.real))
        s_im = float(np.std(Z_fit.imag))
    else:
        raise ValueError(f"Unknown SCALE_MODE: {SCALE_MODE}")
    s_re = max(s_re, 1e-12)
    s_im = max(s_im, 1e-12)

    if F_SCALE is None:
        residual0 = make_residual_fn(f_fit, Z_fit, weights_fit, s_re, s_im)(np.log(p0.to_vector()))
        F_SCALE = max(mad(residual0), 1e-6)

    p_opt, results = fit_params_global_local(
        f_fit=f_fit,
        Z_fit=Z_fit,
        p0=p0,
        weights=weights_fit,
        s_re=s_re,
        s_im=s_im,
        n_starts=N_STARTS,
        top_k=TOP_K,
        seed=SEED,
        global_method=GLOBAL_METHOD,
        max_nfev=MAX_NFEV,
        loss=LOSS,
        f_scale=F_SCALE,
        de_maxiter=DE_MAXITER,
        de_popsize=DE_POPSIZE,
    )
    STATS.t_total_fit += time.perf_counter() - t_fit0

    fit_model_eval = STATS.model_eval - m0
    fit_res_calls = STATS.residual_calls - r0
    fit_obj_calls = STATS.objective_calls - o0

    print("\n===== Optimized parameters =====")
    for k, v in p_opt.as_dict().items():
        print(f"{k:8s} = {v:.6g}")
    print("\n===== Top-5 candidates (by cost) =====")
    for i, r in enumerate(results[:5], 1):
        print(f"{i:2d}) cost={r['cost']:.6g}, nfev={r['nfev']}, status={r['status']}")
    print("\n===== Complexity stats (fit only) =====")
    print(f"p = {N_PARAMS}")
    print(f"N_freq_fit = {f_fit.size}, N_residual_dim = {2 * f_fit.size}")
    print(f"model_eval = {fit_model_eval}")
    print(f"residual_calls = {fit_res_calls}, objective_calls = {fit_obj_calls}")
    print(f"T_fit_total = {STATS.t_total_fit:.3f} s")
    print(f"T_global = {STATS.t_global:.3f} s, T_local = {STATS.t_local:.3f} s")
    print(
        f"n_starts = {N_STARTS}, top_k = {TOP_K}, de_popsize = {DE_POPSIZE}, de_maxiter = {DE_MAXITER}"
    )

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

    # ---- AIC/BIC on full data ----
    weights_all = compute_freq_weights(
        f_all, Z_all, mode=WEIGHT_MODE, min_w=WEIGHT_MIN, max_w=WEIGHT_MAX, power=WEIGHT_POWER
    )
    if SCALE_MODE == "mad":
        s_re_all = max(mad(Z_all.real), 1e-12)
        s_im_all = max(mad(Z_all.imag), 1e-12)
    else:
        s_re_all = max(float(np.std(Z_all.real)), 1e-12)
        s_im_all = max(float(np.std(Z_all.imag)), 1e-12)
    residual_all = make_residual_fn(f_all, Z_all, weights_all, s_re_all, s_im_all)(np.log(p_opt.to_vector()))
    rss = float(np.dot(residual_all, residual_all))
    n = residual_all.size
    aic, bic = compute_aic_bic(rss, n, N_PARAMS)
    print(f"\nAIC = {aic:.3f}, BIC = {bic:.3f}, RSS = {rss:.6g}, n = {n}")

    # ---- optional validation split ----
    if DO_VAL:
        train_idx, val_idx = block_split_indices(f_fit.size, VAL_BLOCKS, VAL_HOLDOUT, seed=SEED)
        f_train = f_fit[train_idx]
        Z_train = Z_fit[train_idx]
        weights_train = weights_fit[train_idx]
        if SCALE_MODE == "mad":
            s_re_tr = max(mad(Z_train.real), 1e-12)
            s_im_tr = max(mad(Z_train.imag), 1e-12)
        else:
            s_re_tr = max(float(np.std(Z_train.real)), 1e-12)
            s_im_tr = max(float(np.std(Z_train.imag)), 1e-12)

        p_train, _ = fit_params_global_local(
            f_fit=f_train,
            Z_fit=Z_train,
            p0=p0,
            weights=weights_train,
            s_re=s_re_tr,
            s_im=s_im_tr,
            n_starts=N_STARTS,
            top_k=TOP_K,
            seed=SEED,
            global_method=GLOBAL_METHOD,
            max_nfev=MAX_NFEV,
            loss=LOSS,
            f_scale=F_SCALE,
            de_maxiter=DE_MAXITER,
            de_popsize=DE_POPSIZE,
        )
        f_val = f_fit[val_idx]
        Z_val = Z_fit[val_idx]
        weights_val = weights_fit[val_idx]
        residual_val = make_residual_fn(f_val, Z_val, weights_val, s_re_tr, s_im_tr)(np.log(p_train.to_vector()))
        rss_val = float(np.dot(residual_val, residual_val))
        print(f"Validation RSS (block split): {rss_val:.6g} (n={residual_val.size})")

    # ---- GP residual analysis ----
    gp_residual_analysis(f_all, Z_all, p_opt, out_prefix="exp_10_gp_residual")



if __name__ == "__main__":
    main()
