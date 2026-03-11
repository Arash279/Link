# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0) Utilities
# ============================================================

def wrap_phase_deg(phi_deg: np.ndarray) -> np.ndarray:
    return (phi_deg + 180.0) % 360.0 - 180.0

def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * np.deg2rad(phase_deg))

def complex_to_mag_phase(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.abs(Z)
    phase = wrap_phase_deg(np.angle(Z, deg=True))
    return mag, phase

def safe_log10(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, float), eps))


# ============================================================
# 1) Load experiment data (Freq, Zabs, Phase) from SQLite
# ============================================================

def load_experiment_from_db(
    db_path: str,
    table: str,
    f_min: float = 10.0,
    f_max: float = 1e8,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT Freq, Zabs, Phase FROM {table}"
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    df = df.dropna().copy()
    df = df.sort_values("Freq")
    df = df[(df["Freq"] >= f_min) & (df["Freq"] <= f_max)]
    return df.reset_index(drop=True)


# ============================================================
# 2) MLP-2 split: cross-band extrapolation
# ============================================================

def make_crossband_split(
    f_all: np.ndarray,
    f_split: float = 1e7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train:  f <= f_split
    Test:   f >  f_split
    """
    idx = np.arange(len(f_all))
    train_idx = idx[f_all <= f_split]
    test_idx  = idx[f_all >  f_split]
    return train_idx, test_idx


# ============================================================
# 3) Dataset pack
# ============================================================

@dataclass
class DatasetPack:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    f_train: np.ndarray
    f_test: np.ndarray
    Z_train: np.ndarray
    Z_test: np.ndarray


def build_dataset_for_mlp(
    f_all: np.ndarray,
    Z_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> DatasetPack:
    x = safe_log10(f_all).reshape(-1, 1)
    y = np.column_stack([Z_all.real, Z_all.imag])

    return DatasetPack(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        f_train=f_all[train_idx],
        f_test=f_all[test_idx],
        Z_train=Z_all[train_idx],
        Z_test=Z_all[test_idx],
    )


# ============================================================
# 4) Metrics (raw-space + readable mag/phase errors)
# ============================================================

def evaluate_complex_prediction(Z_pred: np.ndarray, Z_true: np.ndarray, p: int = 0) -> Dict[str, float]:
    err_re = Z_pred.real - Z_true.real
    err_im = Z_pred.imag - Z_true.imag

    sse = float(np.sum(err_re**2 + err_im**2))
    n = 2 * len(Z_true)
    rmse = float(np.sqrt(sse / max(n, 1)))

    sigma2 = sse / max(n, 1)
    if sigma2 <= 0:
        aic = np.inf
        bic = np.inf
    else:
        aic = float(n * np.log(sigma2) + 2 * p)
        bic = float(n * np.log(sigma2) + p * np.log(n))

    mag_pred, ph_pred = complex_to_mag_phase(Z_pred)
    mag_true, ph_true = complex_to_mag_phase(Z_true)

    logmag_mae = float(np.mean(np.abs(safe_log10(mag_pred) - safe_log10(mag_true))))

    dph = ph_pred - ph_true
    dph = (dph + 180.0) % 360.0 - 180.0
    phase_mae = float(np.mean(np.abs(dph)))

    return {
        "SSE_raw": sse,
        "RMSE_raw": rmse,
        "AIC_raw": aic,
        "BIC_raw": bic,
        "logmag_MAE": logmag_mae,
        "phase_MAE_deg": phase_mae,
        "n": n,
        "p": p,
    }


# ============================================================
# 5) MLP model
# ============================================================

@dataclass
class MLPBundle:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    model: MLPRegressor


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 128, 64),
    random_state: int = 0,
    max_iter: int = 4000,
) -> MLPBundle:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_s = x_scaler.fit_transform(x_train)
    y_train_s = y_scaler.fit_transform(y_train)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=80,
        random_state=random_state,
    )
    model.fit(x_train_s, y_train_s)

    return MLPBundle(x_scaler=x_scaler, y_scaler=y_scaler, model=model)


def predict_mlp(bundle: MLPBundle, x: np.ndarray) -> np.ndarray:
    x_s = bundle.x_scaler.transform(x)
    y_pred_s = bundle.model.predict(x_s)
    y_pred = bundle.y_scaler.inverse_transform(y_pred_s)
    return y_pred


# ============================================================
# 6) Plotting
# ============================================================

def plot_crossband_result(
    f_all: np.ndarray,
    Z_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Z_pred_test: np.ndarray,
    f_split: float,
    title_suffix: str = "",
):
    mag_all, ph_all = complex_to_mag_phase(Z_all)
    f_train = f_all[train_idx]
    Z_train = Z_all[train_idx]
    mag_train, ph_train = complex_to_mag_phase(Z_train)

    f_test = f_all[test_idx]
    mag_pred, ph_pred = complex_to_mag_phase(Z_pred_test)

    plt.figure(figsize=(12, 8))

    # Magnitude (log10|Z|)
    plt.subplot(2, 1, 1)
    plt.semilogx(f_all, safe_log10(mag_all), label="Experiment (all)", linewidth=2)
    # Train points: smaller and red (match MLP_1 style)
    plt.semilogx(f_train, safe_log10(mag_train), "o", markersize=2, color="red", label="Train band points")
    # MLP predictions on test: orange (match MLP_1 style)
    plt.semilogx(f_test, safe_log10(mag_pred), ".", markersize=3, color="orange", label="MLP pred (test band)")
    plt.axvline(f_split, linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(f"MLP-2 Cross-band Extrapolation: Magnitude {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    # Phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f_all, ph_all, label="Experiment (all)", linewidth=2)
    # Train points: smaller and red (match MLP_1 style)
    plt.semilogx(f_train, ph_train, "o", markersize=2, color="red", label="Train band points")
    # MLP predictions on test: orange (match MLP_1 style)
    plt.semilogx(f_test, ph_pred, ".", markersize=3, color="orange", label="MLP pred (test band)")
    plt.axvline(f_split, linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"MLP-2 Cross-band Extrapolation: Phase {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Main
# ============================================================

def main():
    # -------------------- user config --------------------
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"

    F_MIN = 10.0
    F_MAX = 1e8
    F_SPLIT = 1e7  # train <= 1e7, test > 1e7

    RANDOM_STATE = 0
    HIDDEN = (128, 128, 64)
    MAX_ITER = 4000
    # ----------------------------------------------------

    df = load_experiment_from_db(DB_PATH, TABLE, f_min=F_MIN, f_max=F_MAX)

    f_all = df["Freq"].to_numpy(dtype=float)
    zabs_all = df["Zabs"].to_numpy(dtype=float)
    phase_all = df["Phase"].to_numpy(dtype=float)

    Z_all = mag_phase_to_complex(zabs_all, phase_all)

    train_idx, test_idx = make_crossband_split(f_all, f_split=F_SPLIT)

    if len(train_idx) < 20 or len(test_idx) < 20:
        raise RuntimeError(
            f"Split too small: N_train={len(train_idx)}, N_test={len(test_idx)}. "
            f"Check F_SPLIT / F_MIN / F_MAX."
        )

    data = build_dataset_for_mlp(f_all, Z_all, train_idx, test_idx)

    bundle = fit_mlp(
        x_train=data.x_train,
        y_train=data.y_train,
        hidden_layer_sizes=HIDDEN,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )

    y_pred_test = predict_mlp(bundle, data.x_test)
    Z_pred_test = y_pred_test[:, 0] + 1j * y_pred_test[:, 1]

    # Raw-space evaluation ONLY on test band (extrapolation band)
    metrics_test = evaluate_complex_prediction(Z_pred_test, data.Z_test, p=0)

    # (Optional) also evaluate on train band to confirm it fits training region
    y_pred_train = predict_mlp(bundle, data.x_train)
    Z_pred_train = y_pred_train[:, 0] + 1j * y_pred_train[:, 1]
    metrics_train = evaluate_complex_prediction(Z_pred_train, data.Z_train, p=0)

    print("\n===== MLP-2: Cross-band extrapolation =====")
    print(f"DB_PATH = {DB_PATH}")
    print(f"TABLE   = {TABLE}")
    print(f"F_SPLIT = {F_SPLIT:.3g} Hz (train <= split, test > split)")
    print(f"N_all   = {len(f_all)}")
    print(f"N_train = {len(train_idx)}")
    print(f"N_test  = {len(test_idx)}")
    print(f"Hidden layers = {HIDDEN}")
    print(f"Iterations used = {bundle.model.n_iter_}")
    print(f"Final training loss = {bundle.model.loss_:.6g}")

    print("\n===== Raw-space evaluation (TRAIN band) =====")
    print(f"RMSE_raw     = {metrics_train['RMSE_raw']:.6g}")
    print(f"logmag_MAE   = {metrics_train['logmag_MAE']:.6g}")
    print(f"phase_MAE_deg= {metrics_train['phase_MAE_deg']:.6g}")

    print("\n===== Raw-space evaluation (TEST band: extrapolation) =====")
    print(f"SSE_raw      = {metrics_test['SSE_raw']:.6g}")
    print(f"RMSE_raw     = {metrics_test['RMSE_raw']:.6g}")
    print(f"AIC_raw      = {metrics_test['AIC_raw']:.6f}")
    print(f"BIC_raw      = {metrics_test['BIC_raw']:.6f}")
    print(f"logmag_MAE   = {metrics_test['logmag_MAE']:.6g}")
    print(f"phase_MAE_deg= {metrics_test['phase_MAE_deg']:.6g}")

    plot_crossband_result(
        f_all=f_all,
        Z_all=Z_all,
        train_idx=train_idx,
        test_idx=test_idx,
        Z_pred_test=Z_pred_test,
        f_split=F_SPLIT,
        title_suffix=f"({TABLE})",
    )


if __name__ == "__main__":
    main()