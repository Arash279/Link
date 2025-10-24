import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ======================================================
# 配置区
# ======================================================
DB_PATH = r"D:\Desktop\data\AP_1p5.db"
TABLES = ["exp_10", "exp_11", "exp_12"]

# 频率范围：低频区间（堵转拟合）
F_MIN = 100.0     # Hz
F_MAX = 500.0     # Hz

# 已知参数（静止 s=1）
Rs  = 8.703        # Ω
Lls = 2.55e-2      # H
Llr = 2.55e-2      # H
# 若要包含励磁支路，可取消下面两行注释：
# Lm  = 3.2e-2      # H
# Rfe = 1e4         # Ω

# ======================================================
# 函数定义
# ======================================================
def read_data(db_path, table, fmin, fmax):
    """从 SQLite 表中读取频率、幅值、相位并生成复阻抗"""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT Freq, Zabs, Phase FROM {table}", conn)
    df = df.dropna()
    df = df[(df["Freq"] >= fmin) & (df["Freq"] <= fmax)]
    if df.empty:
        raise ValueError(f"{table}: 数据为空或频率范围无效。")
    f = df["Freq"].to_numpy()
    zabs = df["Zabs"].to_numpy()
    phi = np.deg2rad(df["Phase"].to_numpy())
    z = zabs * (np.cos(phi) + 1j * np.sin(phi))
    return f, z

def Z_model(f, Rr):
    """简化等效电路模型（忽略励磁支路）"""
    omega = 2 * np.pi * f
    return Rs + 1j*omega*Lls + Rr + 1j*omega*Llr

def residual(Rr, f, Z_meas):
    """复阻抗残差模"""
    return np.abs(Z_meas - Z_model(f, Rr))

def fit_Rr(f, Z_meas):
    """最小二乘拟合 Rr"""
    res = least_squares(lambda R: residual(R, f, Z_meas),
                        x0=[5.0], bounds=(0, 100))
    return res.x[0]

# ======================================================
# 主程序
# ======================================================
if __name__ == "__main__":
    results = []
    print("=== Rotor Resistance Fitting (Blocked-rotor, s=1) ===\n")

    for tb in TABLES:
        try:
            f, Z_meas = read_data(DB_PATH, tb, F_MIN, F_MAX)
            Rr_fit = fit_Rr(f, Z_meas)
            # 计算拟合残差均方误差
            Z_fit = Z_model(f, Rr_fit)
            mse = np.mean(np.abs(Z_meas - Z_fit)**2)
            print(f"{tb}:  Fitted Rr = {Rr_fit:.4f} Ω,  MSE = {mse:.4e}")
            results.append(Rr_fit)
        except Exception as e:
            print(f"[WARN] {tb}: 拟合失败 -> {e}")

    # 汇总结果
    if results:
        mean_Rr = np.mean(results)
        std_Rr  = np.std(results)
        print("\n=== Summary ===")
        for tb, val in zip(TABLES, results):
            print(f"{tb:8s} : Rr = {val:.4f} Ω")
        print(f"\nAverage Rr = {mean_Rr:.4f} Ω")
        print(f"Std Dev    = {std_Rr:.4f} Ω")
        print(f"Range      = [{min(results):.4f}, {max(results):.4f}] Ω")
    else:
        print("\nNo valid results. 请检查数据库或频率区间。")

    print("\nDone.")
