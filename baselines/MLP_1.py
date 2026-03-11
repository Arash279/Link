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
    """Wrap phase to (-180, 180]."""
    return (phi_deg + 180.0) % 360.0 - 180.0


def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Convert magnitude + phase(deg) to complex impedance."""
    return mag * np.exp(1j * np.deg2rad(phase_deg))


def complex_to_mag_phase(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert complex impedance to magnitude and wrapped phase."""
    mag = np.abs(Z)
    phase = wrap_phase_deg(np.angle(Z, deg=True))
    return mag, phase


def safe_log10(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, float), eps))


# ============================================================
# 1) Load experiment data from SQLite
#    (same column assumptions as your Only_fit.py)
# ============================================================

def load_experiment_from_db(
    db_path: str,
    table: str = "exp_10",
    f_min: float = 10.0,
    f_max: float = 1e8,
) -> pd.DataFrame:
    """
    Expect columns: Freq, Zabs, Phase
    Keep frequencies within [f_min, f_max].
    """
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
# 2) MLP-1 split: in-band interpolation
#    Train on sparse points across whole band, test on the rest
# ============================================================

def make_inband_interpolation_split(
    f_all: np.ndarray,
    stride: int = 5,
    offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example:
      stride=5 means every 5th point is used for training,
      the remaining points are used for test.
    This keeps train/test both inside the full frequency band.
    """
    n = len(f_all)
    idx = np.arange(n)

    train_idx = idx[offset::stride]
    test_idx = np.setdiff1d(idx, train_idx, assume_unique=True)

    return train_idx, test_idx


# ============================================================
# 3) Feature / target preparation
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
    """
    Input feature:
      x = log10(f)

    Target:
      y = [Re(Z), Im(Z)]
    """
    logf = safe_log10(f_all).reshape(-1, 1)

    y = np.column_stack([Z_all.real, Z_all.imag])

    x_train = logf[train_idx]
    y_train = y[train_idx]

    x_test = logf[test_idx]
    y_test = y[test_idx]

    return DatasetPack(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        f_train=f_all[train_idx],
        f_test=f_all[test_idx],
        Z_train=Z_all[train_idx],
        Z_test=Z_all[test_idx],
    )


# ============================================================
# 4) Metrics in raw space
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

    # also report mag/phase errors for readability
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


def fit_mlp_interpolator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 128, 64),
    random_state: int = 0,
    max_iter: int = 3000,
) -> MLPBundle:
    """
    A small but decent MLP for 1D frequency -> 2D [Re, Im].
    """
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
        n_iter_no_change=50,
        random_state=random_state,
    )
    model.fit(x_train_s, y_train_s)

    return MLPBundle(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model=model,
    )


def predict_mlp(bundle: MLPBundle, x: np.ndarray) -> np.ndarray:
    x_s = bundle.x_scaler.transform(x)
    y_pred_s = bundle.model.predict(x_s)
    y_pred = bundle.y_scaler.inverse_transform(y_pred_s)
    return y_pred


# ============================================================
# 6) Plotting
# ============================================================

def plot_mlp_inband_result(
    f_all: np.ndarray,
    Z_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Z_pred_test: np.ndarray,
    title_suffix: str = "",
):
    """
    Plot:
    - full experiment curve
    - train points
    - MLP prediction on test points
    """
    mag_all, ph_all = complex_to_mag_phase(Z_all)
    mag_pred, ph_pred = complex_to_mag_phase(Z_pred_test)

    f_train = f_all[train_idx]
    Z_train = Z_all[train_idx]
    mag_train, ph_train = complex_to_mag_phase(Z_train)

    f_test = f_all[test_idx]

    plt.figure(figsize=(12, 8))

    # magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f_all, safe_log10(mag_all), label="Experiment (all)", linewidth=2)
    # Train points: smaller and red
    plt.semilogx(f_train, safe_log10(mag_train), "o", markersize=2, color="red", label="Train points")
    # MLP predictions on test: orange
    plt.semilogx(f_test, safe_log10(mag_pred), ".", markersize=3, color="orange", label="MLP pred (test)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(f"MLP-1 In-band Interpolation: Magnitude {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    # phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f_all, ph_all, label="Experiment (all)", linewidth=2)
    # Train points: smaller and red
    plt.semilogx(f_train, ph_train, "o", markersize=2, color="red", label="Train points")
    # MLP predictions on test: orange
    plt.semilogx(f_test, ph_pred, ".", markersize=3, color="orange", label="MLP pred (test)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"MLP-1 In-band Interpolation: Phase {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Main
# ============================================================

def main():
    # --------------------------------------------------------
    # User config (keep close to your Only_fit.py style)
    # --------------------------------------------------------
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"

    F_MIN = 10.0
    F_MAX = 1e8

    # MLP-1: in-band interpolation
    STRIDE = 5      # every 5th point for training
    OFFSET = 0      # can change to 1,2,3,4 for repeated runs

    RANDOM_STATE = 0
    HIDDEN = (128, 128, 64)
    MAX_ITER = 3000

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = load_experiment_from_db(DB_PATH, TABLE, f_min=F_MIN, f_max=F_MAX)

    f_all = df["Freq"].to_numpy(dtype=float)
    zabs_all = df["Zabs"].to_numpy(dtype=float)
    phase_all = df["Phase"].to_numpy(dtype=float)

    Z_all = mag_phase_to_complex(zabs_all, phase_all)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    train_idx, test_idx = make_inband_interpolation_split(
        f_all=f_all,
        stride=STRIDE,
        offset=OFFSET,
    )

    data = build_dataset_for_mlp(
        f_all=f_all,
        Z_all=Z_all,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    # --------------------------------------------------------
    # Fit MLP
    # --------------------------------------------------------
    bundle = fit_mlp_interpolator(
        x_train=data.x_train,
        y_train=data.y_train,
        hidden_layer_sizes=HIDDEN,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )

    # --------------------------------------------------------
    # Predict on test set
    # --------------------------------------------------------
    y_pred_test = predict_mlp(bundle, data.x_test)
    Z_pred_test = y_pred_test[:, 0] + 1j * y_pred_test[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = evaluate_complex_prediction(
        Z_pred=Z_pred_test,
        Z_true=data.Z_test,
        p=0,   # black-box surrogate, here we do not count circuit parameters
    )

    print("\n===== MLP-1: In-band interpolation =====")
    print(f"DB_PATH = {DB_PATH}")
    print(f"TABLE   = {TABLE}")
    print(f"N_all   = {len(f_all)}")
    print(f"N_train = {len(train_idx)}")
    print(f"N_test  = {len(test_idx)}")
    print(f"Train stride = {STRIDE}, offset = {OFFSET}")
    print(f"Hidden layers = {HIDDEN}")
    print(f"Iterations used = {bundle.model.n_iter_}")
    print(f"Final training loss = {bundle.model.loss_:.6g}")

    print("\n===== Raw-space evaluation on TEST points =====")
    print(f"SSE_raw      = {metrics['SSE_raw']:.6g}")
    print(f"RMSE_raw     = {metrics['RMSE_raw']:.6g}")
    print(f"AIC_raw      = {metrics['AIC_raw']:.6f}")
    print(f"BIC_raw      = {metrics['BIC_raw']:.6f}")
    print(f"logmag_MAE   = {metrics['logmag_MAE']:.6g}")
    print(f"phase_MAE_deg= {metrics['phase_MAE_deg']:.6g}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_mlp_inband_result(
        f_all=f_all,
        Z_all=Z_all,
        train_idx=train_idx,
        test_idx=test_idx,
        Z_pred_test=Z_pred_test,
        title_suffix=f"({TABLE})"
    )


if __name__ == "__main__":
    main()