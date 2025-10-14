import sqlite3
import numpy as np

DB_PATH = r"D:\Desktop\motor-impedance-db-main\AP_1p5.db"
SINGLE_TABLES = [f"exp_{i}" for i in range(1, 7)]
CM_TABLES = ["exp_13", "exp_17"]
DELTA_TABLE = "exp_21"

# 平台与稳健性参数
PHASE_THRESH_DEG_1, CV_MAX_1, MIN_DECADE_1 = -80.0, 0.05, 0.5
PHASE_THRESH_DEG_2, CV_MAX_2, MIN_DECADE_2 = -75.0, 0.10, 0.5
MIN_POINTS, IQR_K = 10, 1.5

# 与谐振相关的“安全边距”（按频率倍数）：
# LF 平台只在 f <= f_res / RES_MARGIN，HF 平台只在 f >= f_res * RES_MARGIN
RES_MARGIN = 1.2  # ≈ 0.079 decade

def fetch_table(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
    arr = np.array(cur.fetchall(), dtype=float)
    arr = arr[(arr[:,0]>0)&(arr[:,1]>0)]
    arr = arr[np.argsort(arr[:,0])]
    return arr[:,0], arr[:,1], arr[:,2]

def to_Cp(f, mag, th_deg):
    th = np.deg2rad(th_deg)
    Z = mag*(np.cos(th)+1j*np.sin(th))
    Y = 1/Z
    return np.imag(Y)/(2*np.pi*f)

def iqr(x): q75,q25=np.percentile(x,[75,25]); return q75-q25
def cv(x): med=np.median(x); return np.std(x,ddof=1)/abs(med) if med!=0 else np.inf

def robust_mask(x, k=IQR_K):
    """与 x 同长的掩码：|x - median| <= k*IQR 且为有限值。"""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        return np.zeros_like(x, dtype=bool)
    med = np.median(xf)
    iq = iqr(xf)
    if not np.isfinite(med) or iq == 0:
        return finite
    lo, hi = med - k * iq, med + k * iq
    return (x >= lo) & (x <= hi) & finite

def summarize_Cp(Cp, slc):
    if slc is None: return np.nan, np.nan
    x = Cp[slc][robust_mask(Cp[slc])]
    med = float(np.median(x))
    err = float(iqr(x)/med) if med != 0 else np.inf
    return med, err

def check_hf_capacitive(th, tail_ratio=0.2, thr=PHASE_THRESH_DEG_1):
    k = max(5, int(len(th)*tail_ratio))
    tail = th[-k:]
    return np.mean(tail <= thr), k

def print_window_info(label, f, slc):
    if slc is None:
        print(f"  {label} 窗口：未找到")
        return
    i0, i1 = slc.start, slc.stop-1
    f0, f1 = f[i0], f[i1]
    decade_span = np.log10(f1) - np.log10(f0)
    n_pts = slc.stop - slc.start
    print(f"  {label} 窗口：[{f0:.3g} Hz, {f1:.3g} Hz], span≈{decade_span:.3f} decade, 点数={n_pts}")

# —— 核心：用相位对数导数找“第一处谐振边沿”频率 f_res ——
def first_resonance_freq(f, theta_deg):
    logf = np.log10(f)
    dth = np.diff(theta_deg)
    dlf = np.diff(logf)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.abs(dth / dlf)
    slope[~np.isfinite(slope)] = 0.0
    # 只看“首次显著峰”，避免高频尾部乱跳：取前 70% 频段的最大值
    cut = max(5, int(0.7 * slope.size))
    idx = np.argmax(slope[:cut])  # 索引对应区间 [idx, idx+1]
    # 用区间中点的 logf 反推频率
    f_res = 10 ** ((logf[idx] + logf[idx+1]) / 2.0)
    return f_res

# —— 只在“允许的频段”里找平台：LF=谐振前；HF=谐振后 ——
def find_platform_segmented(f, Cp, th,
                            kind="LF",
                            phase_thresh_1=PHASE_THRESH_DEG_1,
                            cv_max_1=CV_MAX_1,
                            min_decade_1=MIN_DECADE_1,
                            phase_thresh_2=PHASE_THRESH_DEG_2,
                            cv_max_2=CV_MAX_2,
                            min_decade_2=MIN_DECADE_2):
    """
    kind: "LF" 或 "HF"
    在限定频段内（LF: f <= f_res/RES_MARGIN；HF: f >= f_res*RES_MARGIN 且 θ<=阈值）寻找最宽平台
    """
    N = len(f)
    if N < MIN_POINTS:
        return None

    f_res = first_resonance_freq(f, th)

    if kind == "LF":
        allowed = f <= (f_res / RES_MARGIN)
    else:
        allowed = (f >= (f_res * RES_MARGIN)) & (th <= phase_thresh_2)  # HF 相位也要求电容性

    def search_once(phase_thr, cv_max, min_dec):
        best = None
        best_span = -1.0
        logf = np.log10(f)
        mask_phase = (th <= phase_thr) & allowed
        # 双指针按自然索引扫描
        i = 0
        while i < N - MIN_POINTS:
            if not mask_phase[i]:
                i += 1
                continue
            # 快速跳到“允许区段”的开始
            if not allowed[i]:
                i += 1
                continue
            # 尝试扩展 j
            j = i + MIN_POINTS
            while j <= N and allowed[j-1]:
                if np.sum(mask_phase[i:j]) >= MIN_POINTS:
                    span = logf[j-1] - logf[i]
                    if span >= min_dec:
                        Cpwin = Cp[i:j][robust_mask(Cp[i:j])]
                        if len(Cpwin) >= MIN_POINTS and cv(Cpwin) < cv_max:
                            # 尽量扩张
                            j2 = j
                            while j2 <= N and (j2 == N or allowed[j2-1]):
                                if j2 == N:
                                    break
                                span2 = logf[j2] - logf[i]
                                if span2 < min_dec:
                                    j2 += 1
                                    continue
                                Cp_w2 = Cp[i:j2+1][robust_mask(Cp[i:j2+1])]
                                if len(Cp_w2) >= MIN_POINTS and cv(Cp_w2) < cv_max:
                                    j2 += 1
                                else:
                                    break
                            span_final = logf[j2-1] - logf[i]
                            if span_final > best_span:
                                best_span = span_final
                                best = slice(i, j2)
                            break
                j += 1
            # 跳过当前起点
            i += 1
        return best

    # 先用严格阈值；不成再放宽一次
    s = search_once(phase_thresh_1, cv_max_1, min_decade_1)
    if s is None:
        s = search_once(phase_thresh_2, cv_max_2, min_decade_2)
    return s

def main():
    conn = sqlite3.connect(DB_PATH)

    print("=== Step 1. 高频端电容性平台确认（并纠正 LF/HF 区段） ===")
    single = []
    for tb in SINGLE_TABLES:
        f, m, th = fetch_table(conn, tb)
        Cp = to_Cp(f, m, th)

        # ——— 用“谐振前/后”找 LF/HF 平台 ———
        sL = find_platform_segmented(f, Cp, th, kind="LF")
        sH = find_platform_segmented(f, Cp, th, kind="HF")
        cL, eL = summarize_Cp(Cp, sL)
        cH, eH = summarize_Cp(Cp, sH)

        frac, k = check_hf_capacitive(th)
        fmin = f.min() if len(f) else np.inf
        c0 = cL - 3*cH

        print(f"[{tb}] HF≤-80°比例 {frac:.2f}  fmin={fmin:.2f}Hz")
        print_window_info("LF", f, sL)
        print_window_info("HF", f, sH)
        print(f"  Csf^LF={cL:.6e}F err={eL:.3f} | Csf^HF={cH:.6e}F err={eH:.3f}")
        print(f"  Csf0={c0:.6e}F\n")

        single.append((cH, cL, eH, eL))

    medHF = np.median([s[0] for s in single if np.isfinite(s[0])])

    print("=== Step 2. Δ 接校验 ===")
    cm = []
    for tb in CM_TABLES:
        f, m, th = fetch_table(conn, tb)
        Cp = to_Cp(f, m, th)
        s = find_platform_segmented(f, Cp, th, kind="HF")
        c, e = summarize_Cp(Cp, s)
        print(f"[{tb}] C_CM^HF={c:.6e}F err={e:.3f}")
        print_window_info("CM-HF", f, s)
        cm.append((tb, c))
    c13 = next((c for t, c in cm if t == "exp_13"), np.nan)

    try:
        f, m, th = fetch_table(conn, DELTA_TABLE)
        Cp = to_Cp(f, m, th)
        s = find_platform_segmented(f, Cp, th, kind="HF")
        cd, ed = summarize_Cp(Cp, s)
        print(f"[{DELTA_TABLE}] C_CM,Δ^HF={cd:.6e}F err={ed:.3f}")
        print_window_info("Δ-HF", f, s)
    except Exception as e:
        cd = np.nan
        print(f"Δ表读取失败: {e}")

    if np.isfinite(cd) and np.isfinite(c13) and c13 > 0:
        print(f"  Δ/(2×#13)={cd/(2*c13):.3f}")
    if np.isfinite(cd) and np.isfinite(medHF) and medHF > 0:
        print(f"  Δ/(6×single_HF_med)={cd/(6*medHF):.3f}")

    print("=== Step 4. 平台质量汇总（目标 err ≤ 0.3） ===")
    for (tb, (cH, cL, eH, eL)) in zip(SINGLE_TABLES, single):
        print(f"[{tb}] HFerr={eH:.3f}  LFerr={eL:.3f}")

    conn.close()

if __name__ == "__main__":
    main()
