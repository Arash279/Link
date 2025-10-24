import sqlite3
import math
import statistics as stats
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# ====== 配置区 ======
DB_PATH = r"D:\Desktop\data\AP_1p5.db"
ALL_TABLES = [f"exp_{i}" for i in range(1, 22)]
# 主通道（星形短接 DM&CM）：10-12；校验通道（Δ）：18-20
PRIMARY_GROUPS = ["exp_10", "exp_11", "exp_12"]
CHECK_GROUPS   = ["exp_18", "exp_19", "exp_20"]

# 已知定子串联漏感 Lls
LLS = 2.55e-2  # Henries

F_LO_INIT = 5_000.0
F_HI_INIT = 80_000.0
F_LO_MIN  = 2_000.0
F_HI_MAX  = 100_000.0


# 励磁主导筛选阈值
SLOPE_MIN = -1.2  # d log|BM| / d log f
SLOPE_MAX = -0.8
IM_OVER_RE_RATIO = 2.0  # |Im(Y)| >= 2*|Re(Y)|
LM_CV_MAX = 0.15  # 变异系数阈值（容忍 15%）
MIN_POINTS = 20   # 最少合格点数，避免偶然点


@dataclass
class BandResult:
    Lm_median: float
    Lm_mean: float
    Lm_std: float
    Lm_iqr: float
    n_points: int
    f_lo: float
    f_hi: float
    group: str


def load_table(db_path: str, table: str) -> pd.DataFrame:
    """
    读取一张表，列名必须是 Freq, Zabs, Phase（Phase 单位：度）
    返回 DataFrame，按频率升序并清洗异常值
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT Freq, Zabs, Phase FROM {table}", conn)

    # 清洗数据：去除空值、非正频率/幅值
    df = df.dropna()
    df = df[(df["Freq"] > 0) & (df["Zabs"] > 0)]
    # 排序
    df = df.sort_values("Freq").reset_index(drop=True)
    return df


def complex_impedance_from_abs_phase(zabs: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """
    由幅值与相位（度）还原复阻抗 Z = |Z| * (cos φ + j sin φ)
    """
    phi = np.deg2rad(phase_deg)
    return zabs * (np.cos(phi) + 1j * np.sin(phi))


def derivative_loglog(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    计算 d log|y| / d log x 的局部导数（用中心差分），边界用一阶差分
    """
    # 保护：去除非正/NaN
    mask = (np.isfinite(y)) & (np.isfinite(x)) & (y > 0) & (x > 0)
    y = y.copy()
    x = x.copy()
    if mask.sum() < 3:
        return np.array([np.nan] * len(x))
    yy = np.log(np.abs(y[mask]))
    xx = np.log(x[mask])

    # 数值导数
    dydx = np.empty_like(xx)
    dydx[1:-1] = (yy[2:] - yy[:-2]) / (xx[2:] - xx[:-2])
    dydx[0] = (yy[1] - yy[0]) / (xx[1] - xx[0])
    dydx[-1] = (yy[-1] - yy[-2]) / (xx[-1] - xx[-2])

    # 放回原长度
    out = np.full_like(x, np.nan, dtype=float)
    out[np.where(mask)[0]] = dydx
    return out


def pick_band_by_criteria(
    f: np.ndarray,
    Ymag: np.ndarray,
    slope: np.ndarray,
    f_lo_init: float,
    f_hi_init: float,
    f_lo_min: float,
    f_hi_max: float,
    slope_min: float,
    slope_max: float,
    im_over_re_ratio: float,
    lm_cv_max: float,
    min_points: int,
    Lls: float
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    """
    在起始窗内先筛，若无合格则扩张窗（先向低频到 f_lo_min，再向高频到 f_hi_max）。
    返回 (f_sel, Ymag_sel, (f_lo_used, f_hi_used))；若失败返回 None。
    """
    # 封装一个选择函数
    def select_in_window(f_lo, f_hi):
        mask0 = (f >= f_lo) & (f <= f_hi)
        if mask0.sum() < min_points:
            return None

        Ym = Ymag[mask0]
        sl = slope[mask0]
        fwin = f[mask0]

        # 物理判据
        good = (
            np.isfinite(sl) &
            (sl >= slope_min) & (sl <= slope_max) &
            (np.abs(np.imag(Ym)) >= im_over_re_ratio * np.abs(np.real(Ym)))
        )

        if good.sum() < min_points:
            return None

        # 计算 Lm(f) 并验证稳定性
        omega = 2 * np.pi * fwin[good]
        BM = np.imag(Ym[good])
        # 防护：避免除 0
        valid = (np.abs(BM) > 0)
        if valid.sum() < min_points:
            return None

        Lm_f = 1.0 / (omega[valid] * np.abs(BM[valid]))
        if len(Lm_f) < min_points:
            return None

        # 稳定性判据（CV）
        mu = np.median(Lm_f)
        if mu <= 0:
            return None
        cv = np.std(Lm_f) / mu
        if cv > lm_cv_max:
            return None

        # 通过
        return fwin[good][valid], Ym[good][valid], (f_lo, f_hi)

    # 1) 初始窗
    res = select_in_window(f_lo_init, f_hi_init)
    if res is not None:
        return res

    # 2) 向低频扩到 f_lo_min
    res = select_in_window(f_lo_min, f_hi_init)
    if res is not None:
        return res

    # 3) 向高频扩到 f_hi_max
    res = select_in_window(f_lo_init, f_hi_max)
    if res is not None:
        return res

    # 4) 全范围（最后尝试）
    res = select_in_window(f_lo_min, f_hi_max)
    return res


def compute_group_Lm(
    db_path: str,
    table: str,
    Lls: float,
    f_lo_init: float = F_LO_INIT,
    f_hi_init: float = F_HI_INIT
) -> Optional[BandResult]:
    """
    对单个接线组（表）计算 Lm。
    返回 BandResult 或 None（失败）
    """
    df = load_table(db_path, table)
    if df.empty:
        print(f"[WARN] {table}: empty table.")
        return None

    f = df["Freq"].to_numpy(dtype=float)
    Zabs = df["Zabs"].to_numpy(dtype=float)
    phase = df["Phase"].to_numpy(dtype=float)

    Z_meas = complex_impedance_from_abs_phase(Zabs, phase)

    # 去串联漏抗：Zmag = Zmeas - j ω Lls
    omega = 2 * np.pi * f
    Zmag = Z_meas - 1j * omega * Lls

    # 导纳
    with np.errstate(divide='ignore', invalid='ignore'):
        Ymag = 1.0 / Zmag

    # 在全频上计算 slope
    slope = derivative_loglog(np.imag(Ymag), f)

    picked = pick_band_by_criteria(
        f=f,
        Ymag=Ymag,
        slope=slope,
        f_lo_init=f_lo_init,
        f_hi_init=f_hi_init,
        f_lo_min=F_LO_MIN,
        f_hi_max=F_HI_MAX,
        slope_min=SLOPE_MIN,
        slope_max=SLOPE_MAX,
        im_over_re_ratio=IM_OVER_RE_RATIO,
        lm_cv_max=LM_CV_MAX,
        min_points=MIN_POINTS,
        Lls=Lls
    )

    if picked is None:
        print(f"[WARN] {table}: no suitable band found.")
        return None

    f_sel, Ymag_sel, (flo, fhi) = picked
    omega_sel = 2 * np.pi * f_sel
    BM = np.imag(Ymag_sel)
    # 安全判断
    mask = np.isfinite(omega_sel) & np.isfinite(BM) & (np.abs(BM) > 0)
    if mask.sum() < MIN_POINTS:
        print(f"[WARN] {table}: insufficient valid points after mask.")
        return None

    Lm_f = 1.0 / (omega_sel[mask] * np.abs(BM[mask]))
    # 统计
    Lm_median = float(np.median(Lm_f))
    Lm_mean   = float(np.mean(Lm_f))
    Lm_std    = float(np.std(Lm_f))
    q1, q3    = np.percentile(Lm_f, [25, 75])
    Lm_iqr    = float(q3 - q1)

    # 找 |Z| 峰值和谷值（谐振、反谐振）
    Zmag_abs = np.abs(Z_meas)
    f_res = f[np.argmax(Zmag_abs)]  # 反谐振点
    f_dip = f[np.argmin(Zmag_abs[100:])]  # 粗略谐振点，避开直流
    print(f"[INFO] {table}: resonance ≈ {f_dip:.1f} Hz, anti-resonance ≈ {f_res:.1f} Hz")

    return BandResult(
        Lm_median=Lm_median,
        Lm_mean=Lm_mean,
        Lm_std=Lm_std,
        Lm_iqr=Lm_iqr,
        n_points=int(len(Lm_f)),
        f_lo=flo,
        f_hi=fhi,
        group=table
    )


def summarize_across_groups(results: List[BandResult], tag: str) -> Dict[str, float]:
    """
    跨接线组合并指标（中位数为主）
    """
    if not results:
        return {}

    meds = [r.Lm_median for r in results]
    means = [r.Lm_mean for r in results]

    summary = {
        f"{tag}_Lm_median": float(np.median(meds)),
        f"{tag}_Lm_mean": float(np.mean(means)),
        f"{tag}_Lm_spread_IQR": float(np.percentile(meds, 75) - np.percentile(meds, 25)),
        f"{tag}_Lm_std_of_medians": float(np.std(meds)),
        f"{tag}_n_groups": int(len(results)),
    }
    return summary


def main():
    # 计算主通道（10-12）
    primary_results: List[BandResult] = []
    for tb in PRIMARY_GROUPS:
        if tb not in ALL_TABLES:
            print(f"[INFO] Skip {tb} (not in table list).")
            continue
        res = compute_group_Lm(DB_PATH, tb, LLS, F_LO_INIT, F_HI_INIT)
        if res:
            primary_results.append(res)

    # 计算校验通道（18-20）
    check_results: List[BandResult] = []
    for tb in CHECK_GROUPS:
        if tb not in ALL_TABLES:
            print(f"[INFO] Skip {tb} (not in table list).")
            continue
        res = compute_group_Lm(DB_PATH, tb, LLS, F_LO_INIT, F_HI_INIT)
        if res:
            check_results.append(res)

    # 打印明细
    rows = []
    for r in primary_results + check_results:
        rows.append({
            "group": r.group,
            "Lm_median_H": r.Lm_median,
            "Lm_mean_H": r.Lm_mean,
            "Lm_std_H": r.Lm_std,
            "Lm_IQR_H": r.Lm_iqr,
            "n_points": r.n_points,
            "band_lo_Hz": r.f_lo,
            "band_hi_Hz": r.f_hi
        })
    df_out = pd.DataFrame(rows).sort_values("group")
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print("\n=== Per-group Lm results (H) ===")
    if not df_out.empty:
        print(df_out.to_string(index=False))
    else:
        print("No valid group results.")

    # 汇总
    primary_summary = summarize_across_groups(primary_results, tag="PRIMARY")
    check_summary   = summarize_across_groups(check_results, tag="CHECK")

    # 一致性检查
    if primary_summary and check_summary:
        primary = primary_summary["PRIMARY_Lm_median"]
        check   = check_summary["CHECK_Lm_median"]
        if primary > 0:
            rel_diff = abs(check - primary) / primary
        else:
            rel_diff = float("nan")
        print("\n=== Summary ===")
        print(f"PRIMARY median Lm (10–12): {primary:.6g} H")
        print(f"CHECK   median Lm (18–20): {check:.6g} H")
        print(f"Relative difference: {rel_diff:.2%}")
        if rel_diff > 0.15:
            print("WARN: Δ 与 星短 的 Lm 偏差 > 15%，建议复核频段/接线寄生影响。")
    elif primary_summary:
        print("\n=== Summary ===")
        print(f"PRIMARY median Lm (10–12): {primary_summary['PRIMARY_Lm_median']:.6g} H")
    elif check_summary:
        print("\n=== Summary ===")
        print(f"CHECK median Lm (18–20): {check_summary['CHECK_Lm_median']:.6g} H")
    else:
        print("\nNo summary available.")

    # 导出 CSV
    out_csv = r"D:\Desktop\data\AP_1p5_Lm_results.csv"
    try:
        print("\n=== Detailed Results ===")
        print(df_out.to_string(index=False))
        print("\n=== Summary ===")
        for k, v in {**primary_summary, **check_summary}.items():
            print(f"{k:25s}: {v}")
    except Exception as e:
        print(f"[WARN] Failed to save CSV: {e}")


if __name__ == "__main__":
    main()
