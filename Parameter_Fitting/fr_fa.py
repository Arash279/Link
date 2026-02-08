# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import pandas as pd

# ===================== 配置 =====================
DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"

# 只用 DM 接线三组
TABLES_Y_SHORT = [f"exp_{i}" for i in (10, 11, 12)]   # 训练/定参
TABLES_Y_LONG  = [f"exp_{i}" for i in (14, 15, 16)]   # 可选验证（线长）
TABLES_DELTA   = [f"exp_{i}" for i in (18, 19, 20)]   # 可选验证（拓扑缩放）

# 参数（稳健口径）
SMOOTH_WIN       = 7        # |Z| 平滑窗口点数（Y-长更平，7 更稳）
MIN_PEAK_PROM    = 0.04     # 峰/谷显著性阈值（log|Z| 相对比例）
MIN_SEP_DECADE   = 0.25     # fr 与 fa 的最小 log10 间隔
MAX_DEC_AFTER_FR = 0.8      # fa 搜索上限：fr 右侧 ≤0.8 decade（约 6.3×fr）
PHI_CROSS_TGT    = 0.0      # 零相位穿越参考
PHI_CM_LIMIT     = -75.0    # 右侧截止相位（避免强 CM）
SEARCH_HALF_DEC  = 0.25     # fr 搜索窗半宽（decade）；总宽 ~0.5

# —— Delta 专用放宽（更新）——
DELTA_MIN_SEP_DEC     = 0.20   # fr→fa 最小 log 间隔
DELTA_MAX_DEC_AFTERFR = 1.00   # Δ 的全局最大搜索宽度（保留 1.0，必要时再放开）
DELTA_EARLY_BAND_MAX  = 0.80   # ★ 新增：优先只在 fr 右侧 ≤0.80 decade 的“早段”里找
DELTA_PHI_CM_LIMIT    = -85.0  # Δ 的相位阈值
DELTA_MIN_PROM        = 0.010  # Δ 的浅谷显著性阈值（再放宽到 1%）

# —— fa 选择的三重门槛（统一口径）——
FA_PROM_HI     = 0.10     # prominence 高门槛（先用）
FA_PROM_LO     = 0.05     # prominence 低门槛（回退一步）
FA_PHASE_MIN   = -70.0    # 相位门槛下界
FA_PHASE_MAX   = -30.0    # 相位门槛上界
FA_GAP_MIN     = 0.25     # log10(fa/fr) 下限
FA_GAP_MAX     = 0.55     # log10(fa/fr) 上限（早段优先）
FA_FB_WIN_DEC  = 0.10     # 回退时在拐点 ±0.1 decade 窗内取 |Z| 最小

# —— Delta 的容错（仍然更宽松一些，但也先走“早段优先”）——
DELTA_MIN_PROM = 0.010    # Δ: 最低 prominence（浅谷）
DELTA_GAP_HARD_MAX = 1.10 # Δ: 彻底回退时的硬上限（仅最后兜底用）



# ===================== 基础函数 =====================
def fetch_table(conn, table):
    """
    从表读取并处理：去除无效点、按频率排序、合并重复频点（中位数），
    以保证后续梯度/导数计算在严格递增自变量上进行。
    """
    cur = conn.cursor()
    cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
    arr = np.array(cur.fetchall(), dtype=float)
    arr = arr[(arr[:, 0] > 0) & (arr[:, 1] > 0)]
    df = pd.DataFrame(arr, columns=["Freq", "Zabs", "Phase"]).sort_values("Freq")
    df = df.groupby("Freq", as_index=False).median()  # 合并重复频点
    f = df["Freq"].to_numpy()
    z = df["Zabs"].to_numpy()
    th = df["Phase"].to_numpy()
    return f, z, th

def smooth_moving_avg(y, w=SMOOTH_WIN):
    if w <= 1: return y
    w = int(w) + (int(w) % 2 == 0)  # 奇数
    k = (w - 1) // 2
    pad = np.r_[y[:k][::-1], y, y[-k:][::-1]]
    ker = np.ones(w) / w
    return np.convolve(pad, ker, mode="valid")

def log_prominence(series, idx, left, right):
    """
    简化 prominence（用于 log|Z|）：峰值与左右区间最低谷的差的最小值。
    """
    peak = series[idx]
    left_min  = np.min(series[left:idx])   if idx - left  >= 1 else peak
    right_min = np.min(series[idx+1:right]) if right - (idx+1) >= 1 else peak
    return peak - max(left_min, right_min)

def find_first_zero_cross_down(theta):
    """返回首次从正到 <=0 的索引（右端点）。找不到则返回 -1。"""
    th = np.asarray(theta)
    pos = th > PHI_CROSS_TGT
    cross = np.where((pos[:-1]) & (~(th[1:] > PHI_CROSS_TGT)))[0]
    return int(cross[0] + 1) if cross.size else -1

# ===================== fr / fa 检测 =====================
def first_resonance(f, zmag, theta):
    """
    在零相位下穿附近的 ±SEARCH_HALF_DEC（对数域）窗口内寻找“第一个显著峰”；
    若无显著峰，则在同一窗口内用 argmax 兜底（兼容宽顶/平顶峰）。
    返回: fr, Zmax, idx_fr, search_slice
    """
    xlog = np.log10(f)
    z_s  = smooth_moving_avg(zmag, SMOOTH_WIN)
    lz   = np.log(z_s)

    # 1) 窗口中心：零相位下穿；无下穿则用 dθ/dlogf 最小点
    idx_cross = find_first_zero_cross_down(theta)
    if idx_cross < 0:
        dth = np.diff(theta)
        dlf = np.diff(xlog)
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = dth / dlf
        slope[~np.isfinite(slope)] = 0.0
        idx0 = np.argmin(slope) + 1
    else:
        idx0 = idx_cross

    # 2) 构造对数域窗口（注意：在 log 轴上比较！）
    lf0 = xlog[idx0]
    left  = np.searchsorted(xlog, lf0 - SEARCH_HALF_DEC, side="left")
    right = np.searchsorted(xlog, lf0 + SEARCH_HALF_DEC, side="right")
    left  = max(0, left)
    right = min(len(f), right)
    if right - left < 5:
        left = max(0, idx0 - 20)
        right = min(len(f), idx0 + 21)
    win = slice(left, right)

    # 3) 在窗口内找“第一个显著峰”；允许宽顶（>= 判据）
    cand_peaks = []
    span_max = np.max(lz[win]) - np.min(lz[win])
    denom = max(1e-9, span_max)
    for i in range(left + 1, right - 1):
        if (lz[i] >= lz[i - 1]) and (lz[i] >= lz[i + 1]):  # 宽顶容忍
            prom = log_prominence(lz, i, left, right)
            rel_prom = prom / denom
            if rel_prom >= MIN_PEAK_PROM:
                cand_peaks.append(i)

    if cand_peaks:
        idx_fr = int(cand_peaks[0])
    else:
        # —— 兜底：同一窗口内取 argmax ——（保证 fr 在窗口内）
        idx_fr = int(left + np.argmax(lz[left:right]))

    fr = float(f[idx_fr])
    Zmax = float(z_s[idx_fr])
    return fr, Zmax, idx_fr, win

def first_anti_resonance(f, zmag, theta, idx_fr, is_delta=False):
    """
    选 fa 的“三重门槛”（满足即取“最靠近 fr 的那个”）：
      ① 极值门槛：局部最小，prominence ≥ 5–10%（先 10%，无则 5%）
      ② 相位门槛：-70° < θ < -30°
      ③ 间隔门槛：0.25 ≤ log10(fa/fr) ≤ 0.55  （早段优先）

    若 >0.55（如 Δ-20），优先回到更靠左、首次满足①②的浅谷/拐点；
    找不到明显谷时回退：
      A) 用 log|Z|–log f 的二阶导找 fr 右侧首次“陡降→缓降”拐点，
         在其 ±0.1 decade 窗内取 |Z| 最小；
      B) 或相位导数 dθ/dlogf 首次由负转略正处，在其 ±0.1 decade 内取 |Z| 最小。
    最后仍无：Δ 允许放宽到 log 间隔 ≤ 1.10 并用最低 prominence（1%）挑最早谷。
    """
    if idx_fr is None:
        return np.nan, np.nan, None

    f = np.asarray(f);    xlog = np.log10(f)
    z_s = smooth_moving_avg(zmag, SMOOTH_WIN)
    lz  = np.log(z_s)
    fr_log = xlog[idx_fr]

    # —— 构造三个门槛的“早段优先”掩码 ——
    gap = xlog - fr_log
    phase_ok = (theta > FA_PHASE_MIN) & (theta < FA_PHASE_MAX)
    base_early = (np.arange(len(f)) > idx_fr) & (gap >= FA_GAP_MIN) & (gap <= FA_GAP_MAX) & phase_ok

    def _prominence(series, i, L, R):
        valley = series[i]
        left_max  = np.max(series[L:i])   if i-L >= 1 else valley
        right_max = np.max(series[i+1:R]) if R-(i+1) >= 1 else valley
        denom = max(1e-9, (np.max(series[L:R]) - np.min(series[L:R])))
        return (min(left_max, right_max) - valley) / denom

    def _first_valley(mask, prom_hi, prom_lo, allow_ultra_low=False):
        idxs = np.where(mask)[0]
        if idxs.size < 3:
            return None
        L, R = idxs.min(), idxs.max() + 1
        # 先 10%，再 5%（Δ 可降到 1%）
        for thr in (prom_hi, prom_lo, (DELTA_MIN_PROM if (is_delta and allow_ultra_low) else None)):
            if thr is None:
                continue
            for i in range(L+1, R-1):
                if (lz[i] <= lz[i-1]) and (lz[i] <= lz[i+1]):     # 局部最小（非严格）
                    rp = _prominence(lz, i, L, R)
                    if rp >= thr:
                        return i
        return None

    # ——— Step 1：早段内按“三重门槛”找“最靠近 fr 的谷” ———
    idx_fa = _first_valley(base_early, FA_PROM_HI, FA_PROM_LO, allow_ultra_low=True)
    if idx_fa is not None:
        return float(f[idx_fa]), float(z_s[idx_fa]), int(idx_fa)

    # ——— Step 2：早段回退：曲率/相位拐点 + 窗口内 argmin(|Z|) ———
    def _min_in_window(center_idx):
        if center_idx is None:
            return None
        c_log = xlog[center_idx]
        L = np.searchsorted(xlog, c_log - FA_FB_WIN_DEC, side="left")
        R = np.searchsorted(xlog, c_log + FA_FB_WIN_DEC, side="right")
        L = max(L, idx_fr + 1)
        # 仍需满足 gap ≥ 下限与相位门槛
        mask = (np.arange(len(f)) >= L) & (np.arange(len(f)) < R) & (gap >= FA_GAP_MIN) & phase_ok
        idxs = np.where(mask)[0]
        if idxs.size < 1:
            return None
        return int(idxs[np.argmin(z_s[idxs])])

    # 2A) 曲率：二阶导由负转正（陡降→缓降）
    idxs_early = np.where(base_early)[0]
    if idxs_early.size >= 5:
        L0, R0 = idxs_early.min(), idxs_early.max() + 1
        d1 = np.gradient(lz[L0:R0], xlog[L0:R0])
        d2 = np.gradient(d1,          xlog[L0:R0])
        cross = np.where((d2[:-1] < 0) & (d2[1:] >= 0))[0]
        if cross.size:
            cand = L0 + int(cross[0] + 1)
            j = _min_in_window(cand)
            if j is not None:
                return float(f[j]), float(z_s[j]), int(j)

    # 2B) 相位导数：dθ/dlogf 首次由负转略正
    if idxs_early.size >= 5:
        dphi = np.gradient(theta[L0:R0], xlog[L0:R0])
        pcross = np.where((dphi[:-1] < 0) & (dphi[1:] >= 0))[0]
        if pcross.size:
            cand = L0 + int(pcross[0] + 1)
            j = _min_in_window(cand)
            if j is not None:
                return float(f[j]), float(z_s[j]), int(j)

    # ——— Step 3：彻底回退（仅 Δ 才放开到 1.10 decade；仍取最靠左）———
    if is_delta:
        mask_wide = (np.arange(len(f)) > idx_fr) & (gap >= FA_GAP_MIN) & (gap <= DELTA_GAP_HARD_MAX) & phase_ok
        j = _first_valley(mask_wide, FA_PROM_LO, DELTA_MIN_PROM, allow_ultra_low=True)
        if j is not None:
            return float(f[j]), float(z_s[j]), int(j)
        # 最后兜底：在“宽窗口”内直接 argmin(|Z|)，选最靠左
        idxs = np.where(mask_wide)[0]
        if idxs.size >= 1:
            j = int(idxs[np.argmin(z_s[idxs])])
            return float(f[j]), float(z_s[j]), j

    # 无果
    return np.nan, np.nan, None

# ===================== 主流程 =====================
def process_table(conn, table):
    f, z, th = fetch_table(conn, table)
    fr, Zmax, idx_fr, sr = first_resonance(f, z, th)
    # Δ 组放宽：判断表名是否在 TABLES_DELTA
    is_delta = table in TABLES_DELTA
    fa, Zanti, idx_fa = first_anti_resonance(f, z, th, idx_fr, is_delta=is_delta)
    return {
        "table": table,
        "fr": fr, "Zmax": Zmax, "fa": fa, "Zanti": Zanti,
        "fr_idx": (int(idx_fr) if idx_fr is not None else None),
        "fa_idx": (int(idx_fa) if idx_fa is not None else None),
        "search_lo": float(f[sr.start]) if sr else np.nan,
        "search_hi": float(f[sr.stop - 1]) if sr else np.nan
    }

def median_safe(arr):
    vals = [v for v in arr if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan

def main():
    conn = sqlite3.connect(DB_PATH)

    groups = [
        ("Y-短 (训练/定参)", TABLES_Y_SHORT),
        ("Y-长 (验证)",      TABLES_Y_LONG),
        ("Δ (验证/缩放)",    TABLES_DELTA),
    ]

    all_rows = []
    for title, tables in groups:
        if not tables: continue
        print(f"\n=== {title} ===")
        rows = []
        for tb in tables:
            r = process_table(conn, tb)
            rows.append(r)
            print(f"{tb}: fr={r['fr']:.6g} Hz |Z|max={r['Zmax']:.3e} ;  "
                  f"fa={r['fa']:.6g} Hz |Z|_anti={r['Zanti']:.3e} ;  "
                  f"[search {r['search_lo']:.3g}~{r['search_hi']:.3g}]  "
                  f"idx(fr,fa)={r['fr_idx']},{r['fa_idx']}")
        all_rows.extend(rows)

        # 组内中位数（稳健聚合）
        fr_med   = median_safe([r["fr"]    for r in rows])
        Zmax_med = median_safe([r["Zmax"]  for r in rows])
        fa_med   = median_safe([r["fa"]    for r in rows])
        Zan_med  = median_safe([r["Zanti"] for r in rows])

        # 质量提示
        if title.startswith("Y-短") and np.isfinite(fr_med) and np.isfinite(fa_med):
            gap = np.log10(fa_med) - np.log10(fr_med)
            print(f"  → 组内中位: fr={fr_med:.6g} Hz, fa={fa_med:.6g} Hz, gap={gap:.3f} dec")
        else:
            print(f"  → 组内中位: fr={fr_med:.6g} Hz, fa={fa_med:.6g} Hz")

    # 可选：Y vs Δ 的缩放校验（理论 ≈ sqrt(2/3) ≈ 0.816）
    frY_med = median_safe([r["fr"] for r in all_rows if r["table"] in TABLES_Y_SHORT + TABLES_Y_LONG])
    frD_med = median_safe([r["fr"] for r in all_rows if r["table"] in TABLES_DELTA])
    ratio = (frY_med / frD_med) if (np.isfinite(frY_med) and np.isfinite(frD_med) and frD_med > 0) else np.nan
    print("\n=== 缩放校验（可选）===")
    print(f"fr_Y / fr_Δ ≈ {ratio:.3f} （理论 ≈ sqrt(2/3) ≈ 0.816，偏离较大时请回看峰/谷窗口）")

    # 导出明细（可选）
    df = pd.DataFrame(all_rows)
    out_path = r"D:\Desktop\EE5003\data\resonance_report.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已导出: {out_path}")

    conn.close()

if __name__ == "__main__":
    main()

