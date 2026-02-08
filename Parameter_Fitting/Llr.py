# -*- coding: utf-8 -*-
# compute_leakage_Lsigma.py
# 依赖: pandas, numpy

import re
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# ===================== 配置 =====================
DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
FILES_Y_SHORT = [f"exp_{i}" for i in (10, 11, 12)]
FILES_Y_LONG  = [f"exp_{i}" for i in (14, 15, 16)]
FILES_DELTA   = [f"exp_{i}" for i in (18, 19, 20)]

# 列名约定
COL_F, COL_Z, COL_TH = "freq_hz", "mag_ohm", "phase_deg"

# 平台与质量门槛（可按需调整）
PHI_TH_1       = +80.0    # 首轮相位门槛（感性）
PHI_TH_2       = +75.0    # 放宽相位门槛
MIN_LOG_SPAN   = 0.5      # 最小对数频宽（decade）
MAX_CV_1       = 0.05     # 首轮 CV 上限
MAX_CV_2       = 0.10     # 放宽 CV 上限
MIN_POINTS     = 80       # 平台最少点数（可按数据密度调）
RES_MARGIN     = 1.2      # 与首个谐振的“安全边距”：LF 仅 f <= f_res/RES_MARGIN

# 质量过滤门槛（用于报告标记）
QLTY_ERR_MAX   = 0.15
QLTY_SPAN_MIN  = 0.5

# =================================================
# 工具函数
# =================================================
def read_table_one(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
    arr = np.array(cur.fetchall(), dtype=float)
    arr = arr[(arr[:,0]>0) & (arr[:,1]>0)]
    arr = arr[np.argsort(arr[:,0])]
    df = pd.DataFrame(arr, columns=["freq_hz", "mag_ohm", "phase_deg"])
    return df

def to_Leq(freq, mag, phase_deg):
    theta = np.deg2rad(phase_deg)
    # 感性：Im{Z} = |Z| sin(theta)，Leq = Im{Z} / (2*pi*f)
    Z_im = mag * np.sin(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        Leq = Z_im / (2*np.pi*freq)
    return Leq

def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def cv(x):
    med = np.median(x)
    if med == 0: return np.inf
    return np.std(x, ddof=1) / abs(med)

def robust_mask_median_iqr(x, k=1.5):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        return np.zeros_like(x, dtype=bool)
    med = np.median(xf)
    spread = iqr(xf)
    if not np.isfinite(med) or spread == 0:
        return finite
    lo, hi = med - k*spread, med + k*spread
    keep = (x >= lo) & (x <= hi) & finite
    return keep

def first_resonance_freq(freq, phase_deg):
    """用 |dθ/dlog10f| 的首个显著峰估计首个谐振边沿频率"""
    f = np.asarray(freq); th = np.asarray(phase_deg)
    logf = np.log10(f)
    dth = np.diff(th)
    dlf = np.diff(logf)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.abs(dth / dlf)
    slope[~np.isfinite(slope)] = 0.0
    cut = max(5, int(0.7 * slope.size))  # 只看前 70%
    idx = np.argmax(slope[:cut])
    # 取区间中点
    f_res = 10 ** ((logf[idx] + logf[idx+1]) / 2.0)
    return f_res

def find_inductive_window(freq, phase_deg, Leq,
                          phi_th_1=PHI_TH_1, phi_th_2=PHI_TH_2,
                          min_log_span=MIN_LOG_SPAN,
                          max_cv_1=MAX_CV_1, max_cv_2=MAX_CV_2,
                          min_points=MIN_POINTS):
    """
    在“首个谐振之前”的感性平台内，寻找对数频宽最大的窗口（满足相位、CV、点数）。
    返回 slice(start, end)
    """
    f = np.asarray(freq); th = np.asarray(phase_deg); L = np.asarray(Leq)
    N = f.size
    if N < min_points: return None

    f_res = first_resonance_freq(f, th)
    allowed = f <= (f_res / RES_MARGIN)

    def search_once(phi_th, max_cv):
        best = None; best_span = -1.0
        logf = np.log10(f)
        mask = (th >= phi_th) & allowed
        i = 0
        while i < N - min_points:
            if not mask[i]:
                i += 1; continue
            # 只在“allowed”的连续片段内扩张
            if not allowed[i]:
                i += 1; continue
            j = i + min_points
            while j <= N and allowed[j-1]:
                if np.sum(mask[i:j]) >= min_points:
                    span = logf[j-1] - logf[i]
                    if span >= min_log_span:
                        Lwin = L[i:j][robust_mask_median_iqr(L[i:j])]
                        if Lwin.size >= min_points and cv(Lwin) <= max_cv:
                            # 向右尽量扩
                            j2 = j
                            while j2 < N and allowed[j2]:
                                span2 = logf[j2] - logf[i]
                                if span2 < min_log_span:
                                    j2 += 1; continue
                                Lw2 = L[i:j2+1][robust_mask_median_iqr(L[i:j2+1])]
                                if Lw2.size >= min_points and cv(Lw2) <= max_cv:
                                    j2 += 1
                                else:
                                    break
                            span_final = logf[j2-1] - logf[i]
                            if span_final > best_span:
                                best_span = span_final
                                best = slice(i, j2)
                            break
                j += 1
            i += 1
        return best

    s = search_once(phi_th_1, max_cv_1)
    if s is None:
        s = search_once(phi_th_2, max_cv_2)
    return s

def process_table(conn, table, topology, do_plot=False):
    df = read_table_one(conn, table)
    f  = df["freq_hz"].to_numpy()
    mag = df["mag_ohm"].to_numpy()
    th  = df["phase_deg"].to_numpy()

    Leq_curve = to_Leq(f, mag, th)
    win = find_inductive_window(f, th, Leq_curve)

    if do_plot:
        plot_impedance_with_window(f, mag, th, win, table)

    if win is None:
        return None, {"status": "NO_WINDOW", "file": table}

    i0, i1 = win.start, win.stop - 1
    f_lo, f_hi = float(f[i0]), float(f[i1])
    span_dec = float(np.log10(f_hi) - np.log10(f_lo))
    Lwin = Leq_curve[win][robust_mask_median_iqr(Leq_curve[win])]
    Leq_win = float(np.median(Lwin))
    err = float(iqr(Lwin)/Leq_win) if Leq_win != 0 else np.inf

    # 拓扑换算为“每相泄漏 L_sigma”
    if topology == "Y":
        L_sigma = (2.0/3.0) * Leq_win
    elif topology.lower().startswith("d"):
        L_sigma = 2.0 * Leq_win
    else:
        raise ValueError("topology 必须是 'Y' 或 'Delta'")

    rec = {
        "file": table,                # 用表名标识
        "topology": topology,
        "f_lo": f_lo, "f_hi": f_hi,
        "span_dec": span_dec, "n_pts": int(win.stop - win.start),
        "Leq_win": Leq_win, "L_sigma": float(L_sigma),
        "err": err, "status": "OK"
    }
    return rec, None

def filter_quality(records, err_max=QLTY_ERR_MAX, span_min=QLTY_SPAN_MIN):
    good, bad = [], []
    for r in records:
        if r["status"] != "OK":
            bad.append(r); continue
        ok = (np.isfinite(r["err"]) and r["err"] <= err_max and
              np.isfinite(r["span_dec"]) and r["span_dec"] >= span_min)
        (good if ok else bad).append(r | {"quality_ok": ok})
    return good, bad

def median_safe(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan

import matplotlib.pyplot as plt

def plot_impedance_with_window(f, mag, th_deg, win, table):
    """
    画阻抗幅值(|Z|)与相位(°)，并用阴影标记感性平台窗口 win=slice(start, stop)。
    只展示，不保存。
    """
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)
    th_deg = np.asarray(th_deg, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    # |Z|
    ax1.semilogx(f, mag)
    ax1.set_ylabel("|Z| (Ω)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Phase
    ax2.semilogx(f, th_deg)
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 标记窗口
    if win is not None:
        i0, i1 = win.start, win.stop - 1
        f_lo, f_hi = float(f[i0]), float(f[i1])
        ax1.axvspan(f_lo, f_hi, alpha=0.2)
        ax2.axvspan(f_lo, f_hi, alpha=0.2)
        ax1.axvline(f_lo, linewidth=1.0)
        ax1.axvline(f_hi, linewidth=1.0)
        ax2.axvline(f_lo, linewidth=1.0)
        ax2.axvline(f_hi, linewidth=1.0)

    fig.suptitle(f"{table} |Z| & Phase (inductive window shaded)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.1)  # 给 GUI 事件循环一点时间

# =================================================
# 主流程
# =================================================
def main():
    conn = sqlite3.connect(DB_PATH)
    # 1) 逐文件处理
    rows_Ys, rows_Yl, rows_D = [], [], []
    for fn in FILES_Y_SHORT:
        rec, err = process_table(conn, fn, "Y", do_plot=True)
        rows_Ys.append(rec if rec else err)

    for fn in FILES_Y_LONG:
        rec, err = process_table(conn, fn, "Y", do_plot=True)
        rows_Yl.append(rec if rec else err)

    for fn in FILES_DELTA:
        rec, err = process_table(conn, fn, "Delta", do_plot=True)
        rows_D.append(rec if rec else err)

    # 2) 质量过滤
    good_Ys, bad_Ys = filter_quality(rows_Ys)
    good_Yl, bad_Yl = filter_quality(rows_Yl)
    good_D , bad_D  = filter_quality(rows_D)

    # 3) 组内稳健聚合（中位数）
    Lsig_Ys = median_safe([r["L_sigma"] for r in good_Ys])
    Lsig_Yl = median_safe([r["L_sigma"] for r in good_Yl])
    Lsig_D  = median_safe([r["L_sigma"] for r in good_D])

    # 4) 合并 Y 口径 + 用 Δ 口径交叉验证
    Lsig_Y_combined = median_safe([Lsig_Ys, Lsig_Yl])
    Leq_Y_med = median_safe([r["Leq_win"] for r in good_Ys + good_Yl])
    Leq_D_med = median_safe([r["Leq_win"] for r in good_D])
    ratio_check = (Leq_Y_med / Leq_D_med) if (np.isfinite(Leq_Y_med) and np.isfinite(Leq_D_med) and Leq_D_med != 0) else np.nan
    # 期望 ratio_check ≈ 3～3.3

    # 5) 融合最终泄漏并拆分 L_ls / L_lr
    L_sigma_final = median_safe([Lsig_Y_combined, Lsig_D])  # 简洁稳健：取中位数
    L_ls = L_lr = 0.5 * L_sigma_final

    # 6) 打印报告（含窗口信息）
    def print_block(title, rows):
        print(f"\n=== {title} ===")
        for r in rows:
            if r["status"] != "OK":
                print(f"{r['file']}: {r['status']}")
                continue
            print(f"{r['file']} | win=[{r['f_lo']:.3g},{r['f_hi']:.3g}] Hz  span={r['span_dec']:.3f} dec  "
                  f"n={r['n_pts']:d} | Leq_win={r['Leq_win']:.6e} H  L_sigma={r['L_sigma']:.6e} H  err={r['err']:.3f}")

    print_block("Y-短 (#10,#11,#12)", rows_Ys)
    print_block("Y-长 (#14,#15,#16)", rows_Yl)
    print_block("Δ (#18,#19,#20)", rows_D)

    print("\n=== 汇总 ===")
    print(f"L_sigma(Y-短)  = {Lsig_Ys:.6e} H")
    print(f"L_sigma(Y-长)  = {Lsig_Yl:.6e} H")
    print(f"L_sigma(Δ)     = {Lsig_D:.6e} H")
    print(f"L_sigma(Y合并) = {Lsig_Y_combined:.6e} H")
    print(f"Y:Δ 等效电感比  (Leq_med_Y / Leq_med_Δ) ≈ {ratio_check:.3f}  （期望 3~3.3）")
    print(f"\nL_sigma_final  = {L_sigma_final:.6e} H")
    print(f"L_ls = L_lr    = {L_ls:.6e} H")

    plt.show()
    conn.close()


if __name__ == "__main__":
    main()
