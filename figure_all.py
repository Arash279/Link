import os
import re
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# Config
# ==========================
BASE_DIR = r"D:\Desktop\EE5003\data"
OUT_BASE = os.path.join(BASE_DIR, "figures")

DBS = [
    ("AP_1p5.db", "AP1.5"),
    ("AP_7p5.db", "AP7.5"),
    ("AP_30.db",  "AP30"),
]

# 只处理 exp_数字 这种表名
EXP_TABLE_PATTERN = re.compile(r"^exp_(\d+)$")


def list_exp_tables(conn) -> list[tuple[int, str]]:
    """返回 [(exp编号, 表名), ...]，按编号从小到大排序"""
    df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    tables = []
    for name in df["name"].tolist():
        m = EXP_TABLE_PATTERN.match(name)
        if m:
            tables.append((int(m.group(1)), name))
    tables.sort(key=lambda x: x[0])
    return tables


def load_exp_table(conn, table_name: str) -> pd.DataFrame:
    """读取单个实验表，并做必要清洗/排序"""
    query = f"SELECT Freq, Zabs, Phase FROM {table_name}"
    exp = pd.read_sql_query(query, conn)

    # 转为数值，去掉坏数据
    exp["Freq"] = pd.to_numeric(exp["Freq"], errors="coerce")
    exp["Zabs"] = pd.to_numeric(exp["Zabs"], errors="coerce")
    exp["Phase"] = pd.to_numeric(exp["Phase"], errors="coerce")
    exp = exp.dropna(subset=["Freq", "Zabs", "Phase"])

    # 频率必须 > 0，否则 semilogx 会报错
    exp = exp[exp["Freq"] > 0]

    # Zabs 必须 > 0 才能 log10
    exp = exp[exp["Zabs"] > 0]

    exp = exp.sort_values("Freq")
    return exp


def plot_and_save(exp: pd.DataFrame, title_suffix: str, out_path: str):
    """画两联图并保存"""
    plt.figure(figsize=(12, 8))

    # ----- 上图: log10(|Z|) -----
    plt.subplot(2, 1, 1)
    plt.semilogx(exp["Freq"], np.log10(exp["Zabs"]), linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(f"Impedance Magnitude (log scale) - {title_suffix}")
    plt.grid(True)

    # ----- 下图: Phase -----
    plt.subplot(2, 1, 2)
    plt.semilogx(exp["Freq"], exp["Phase"], linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Impedance Phase - {title_suffix}")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    for db_file, folder_name in DBS:
        db_path = os.path.join(BASE_DIR, db_file)
        out_dir = os.path.join(OUT_BASE, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(db_path):
            print(f"[WARN] DB not found: {db_path}")
            continue

        conn = sqlite3.connect(db_path)
        try:
            exp_tables = list_exp_tables(conn)
            if not exp_tables:
                print(f"[WARN] No exp_* tables found in: {db_path}")
                continue

            for exp_idx, table_name in exp_tables:
                try:
                    exp = load_exp_table(conn, table_name)
                    if exp.empty:
                        print(f"[WARN] Empty/invalid data: {db_file} -> {table_name}")
                        continue

                    out_name = f"exp_{exp_idx}.png"
                    out_path = os.path.join(out_dir, out_name)

                    plot_and_save(exp, f"{db_file} / {table_name}", out_path)
                    print(f"[OK] Saved: {out_path}")

                except Exception as e:
                    print(f"[ERR] Failed: {db_file} -> {table_name} | {e}")

        finally:
            conn.close()


if __name__ == "__main__":
    main()
