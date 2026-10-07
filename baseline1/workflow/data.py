from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_experiment(db_path: str | Path, table: str, max_freq: float | None = None) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(f"SELECT Freq, Zabs, Phase FROM {table}", conn)

    df = df.dropna().copy()
    df = df[(df["Freq"] > 0) & (df["Zabs"] > 0)]
    if max_freq is not None:
        df = df[df["Freq"] <= max_freq]
    return df.sort_values("Freq").reset_index(drop=True)


def frame_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        df["Freq"].to_numpy(dtype=float),
        df["Zabs"].to_numpy(dtype=float),
        df["Phase"].to_numpy(dtype=float),
    )


def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * np.deg2rad(phase_deg))


def sample_freq_points(
    f_all: np.ndarray,
    zabs_all: np.ndarray,
    phase_all: np.ndarray,
    n_samples: int,
    mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_all = np.asarray(f_all)
    n_total = f_all.size
    if n_samples >= n_total:
        return f_all, zabs_all, phase_all

    if mode == "log_uniform":
        logf = np.log10(f_all)
        grid = np.linspace(logf.min(), logf.max(), n_samples)
        idx = np.searchsorted(logf, grid)
        idx = np.clip(idx, 0, n_total - 1)
        idx = np.unique(idx)
        if idx.size < n_samples:
            rng = np.random.default_rng(seed)
            remaining = np.setdiff1d(np.arange(n_total), idx)
            extra = rng.choice(remaining, size=min(n_samples - idx.size, remaining.size), replace=False)
            idx = np.sort(np.concatenate([idx, extra]))
        return f_all[idx], zabs_all[idx], phase_all[idx]

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n_samples, replace=False))
        return f_all[idx], zabs_all[idx], phase_all[idx]

    raise ValueError(f"Unknown sampling mode: {mode}")

