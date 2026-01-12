# -*- coding: utf-8 -*-
"""
exp_1 only: pick resonance (fr, |Z| minimum) and anti-resonance (fa, |Z| maximum)
with a phase~0 criterion (phase-zero crossing neighborhood).

DB schema kept identical:
  SELECT Freq, Zabs, Phase FROM exp_1

Plot:
  - log10(|Z|) vs f (x log scale)
  - phase vs f (x log scale)
  - mark fr (min|Z| near phase 0) and fa (max|Z| near phase 0)
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 配置（沿用口径） =====================
DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
TABLE = "exp_1"

SMOOTH_WIN = 7

# 相位约束：在 0° 附近找极值（避免跑到强 CM 区）
PHI_TGT = 0.0          # 目标相位
PHI_BAND = 20.0        # 允许带宽：|phase-0| <= 20°（可按需要调 10~30）
SEARCH_HALF_DEC = 0.35 # 在相位过0附近的 logf 窗半宽（decade）

# 极值显著性门槛（防止“平台随机点”）
MIN_REL_PROM = 0.02    # 2%（可按数据质量调 0.01~0.05）

# --------------------- 基础 I/O ---------------------
def fetch_table(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT Freq, Zabs, Phase FROM {table}")
    arr = np.array(cur.fetchall(), dtype=float)
    arr = arr[(arr[:, 0] > 0) & (arr[:, 1] > 0) & np.isfinite(arr).all(axis=1)]

    df = pd.DataFrame(arr, columns=["Freq", "Zabs", "Phase"]).sort_values("Freq")
    df = df.groupby("Freq", as_index=False).median()

    f = df["Freq"].to_numpy()
    z = df["Zabs"].to_numpy()
    th = df["Phase"].to_numpy()
    return f, z, th


def smooth_moving_avg(y, w=SMOOTH_WIN):
    if w <= 1:
        return y
    w = int(w) + (int(w) % 2 == 0)  # odd
    k = (w - 1) // 2
    pad = np.r_[y[:k][::-1], y, y[-k:][::-1]]
    ker = np.ones(w) / w
    return np.convolve(pad, ker, mode="valid")


# --------------------- 数值小工具 ---------------------
def local_extrema_indices(y, kind="min"):
    """Local minima/maxima indices (allow flat)."""
    if len(y) < 3:
        return np.array([], dtype=int)
    if kind == "min":
        m = (y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])
    else:
        m = (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:])
    return np.where(m)[0] + 1


def rel_prominence(series, i, left, right, kind):
    """
    Normalized prominence proxy within [left,right):
      - for min: (min(max_left, max_right) - valley) / span
      - for max: (peak - max(min_left, min_right)) / span
    """
    seg = series[left:right]
    span = float(np.max(seg) - np.min(seg) + 1e-12)

    if kind == "min":
        valley = series[i]
        left_max = np.max(series[left:i]) if i > left else valley
        right_max = np.max(series[i+1:right]) if i+1 < right else valley
        prom = min(left_max, right_max) - valley
    else:
        peak = series[i]
        left_min = np.min(series[left:i]) if i > left else peak
        right_min = np.min(series[i+1:right]) if i+1 < right else peak
        prom = peak - max(left_min, right_min)

    return float(prom / span)


def find_first_phase_zero_cross(f, theta):
    """Return index near first 0-crossing (either direction). If none, return argmin |theta|."""
    th = np.asarray(theta)
    for i in range(len(th) - 1):
        if (th[i] <= 0 and th[i+1] > 0) or (th[i] > 0 and th[i+1] <= 0):
            return i + 1
    return int(np.argmin(np.abs(th)))


# --------------------- 核心：在相位过0附近找极小/极大 ---------------------
def pick_fr_fa(f, zmag, theta):
    """
    fr: |Z| local MIN in a neighborhood around phase~0 (series resonance)
    fa: |Z| local MAX in a neighborhood around phase~0 (anti-resonance)

    Steps:
      1) locate phase~0 center (first crossing, else min|theta|)
      2) window in logf: center ± SEARCH_HALF_DEC
      3) within window AND |theta|<=PHI_BAND:
           - pick best local MIN for fr (or fallback argmin)
           - pick best local MAX for fa (or fallback argmax)
         with a weak relative prominence gate MIN_REL_PROM
    """
    f = np.asarray(f); zmag = np.asarray(zmag); theta = np.asarray(theta)
    xlog = np.log10(f)

    z_s = smooth_moving_avg(zmag, SMOOTH_WIN)
    lnz = np.log(z_s)

    idx0 = find_first_phase_zero_cross(f, theta)
    lf0 = xlog[idx0]

    left = np.searchsorted(xlog, lf0 - SEARCH_HALF_DEC, side="left")
    right = np.searchsorted(xlog, lf0 + SEARCH_HALF_DEC, side="right")
    left = max(0, left)
    right = min(len(f), right)

    # phase mask near 0° (hard gate)
    phase_mask = np.abs(theta - PHI_TGT) <= PHI_BAND
    mask = np.zeros_like(phase_mask, dtype=bool)
    mask[left:right] = True
    mask &= phase_mask

    idxs = np.where(mask)[0]
    if idxs.size < 5:
        # too few points -> relax: ignore phase mask but keep window
        mask = np.zeros_like(phase_mask, dtype=bool)
        mask[left:right] = True
        idxs = np.where(mask)[0]

    L = int(idxs.min()); R = int(idxs.max() + 1)

    # --- fr: local minima on ln|Z| ---
    sub = lnz[L:R]
    mins = local_extrema_indices(sub, kind="min") + L
    best_fr = None
    best_score = -1e18
    for i in mins:
        rp = rel_prominence(lnz, i, L, R, kind="min")
        # prefer closer to phase-0 center, but mainly require not-too-flat
        dist = abs(xlog[i] - lf0)
        score = rp - 0.10 * dist
        if rp >= MIN_REL_PROM and score > best_score:
            best_score = score
            best_fr = i
    if best_fr is None:
        best_fr = int(L + np.argmin(z_s[L:R]))

    # --- fa: local maxima on ln|Z| ---
    maxs = local_extrema_indices(sub, kind="max") + L
    best_fa = None
    best_score = -1e18
    for i in maxs:
        rp = rel_prominence(lnz, i, L, R, kind="max")
        dist = abs(xlog[i] - lf0)
        score = rp - 0.10 * dist
        if rp >= MIN_REL_PROM and score > best_score:
            best_score = score
            best_fa = i
    if best_fa is None:
        best_fa = int(L + np.argmax(z_s[L:R]))

    meta = {
        "idx_center": int(idx0),
        "f_center": float(f[idx0]),
        "win_left": int(left),
        "win_right": int(right),
        "effective_L": int(L),
        "effective_R": int(R),
        "phi_band_deg": float(PHI_BAND),
        "search_half_dec": float(SEARCH_HALF_DEC),
        "min_rel_prom": float(MIN_REL_PROM),
    }

    fr = float(f[best_fr])
    fa = float(f[best_fa])
    return fr, best_fr, fa, best_fa, meta


# --------------------- 绘图 ---------------------
def plot_with_marks(f, zmag, theta, fr, idx_fr, fa, idx_fa, title):
    zlog10 = np.log10(zmag)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(f, zlog10, linewidth=1.6)
    ax[0].set_xscale("log")
    ax[0].set_ylabel("log10(|Z|) (Ohm)")
    ax[0].set_title(f"Impedance Magnitude (log10) - {title}")

    ax[1].plot(f, theta, linewidth=1.6)
    ax[1].set_xscale("log")
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_title(f"Impedance Phase - {title}")

    def mark(freq, idx, label):
        for a in ax:
            a.axvline(freq, linestyle="--", linewidth=1.2)
        ax[0].scatter([freq], [zlog10[idx]], s=70)
        ax[1].scatter([freq], [theta[idx]], s=70)
        ax[0].text(freq, zlog10[idx], f"  {label}", va="bottom", ha="left")
        ax[1].text(freq, theta[idx], f"  {label}", va="bottom", ha="left")

    mark(fr, idx_fr, "fr (min|Z|, phase~0)")
    mark(fa, idx_fa, "fa (max|Z|, phase~0)")

    for a in ax:
        a.grid(True, which="both", ls=":")

    plt.tight_layout()
    plt.show()


# --------------------- main ---------------------
def main():
    conn = sqlite3.connect(DB_PATH)
    f, z, th = fetch_table(conn, TABLE)
    conn.close()

    fr, idx_fr, fa, idx_fa, meta = pick_fr_fa(f, z, th)

    print("\n=== exp_1: fr & fa (phase~0 assisted) ===")
    print(f"fr = {fr:.6g} Hz   idx={idx_fr}   |Z|={z[idx_fr]:.3e}   phase={th[idx_fr]:.2f} deg")
    print(f"fa = {fa:.6g} Hz   idx={idx_fa}   |Z|={z[idx_fa]:.3e}   phase={th[idx_fa]:.2f} deg")
    print("--- meta ---")
    for k, v in meta.items():
        print(f"{k}: {v}")

    title = f"{os.path.basename(DB_PATH)} / {TABLE}"
    plot_with_marks(f, z, th, fr, idx_fr, fa, idx_fa, title)


if __name__ == "__main__":
    main()
