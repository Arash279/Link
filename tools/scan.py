from __future__ import annotations

import copy
import sqlite3
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wrap_phase_deg(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    return (phi_deg + 180.0) % 360.0 - 180.0


def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Convert magnitude + phase(deg) to complex."""
    return mag * np.exp(1j * np.deg2rad(phase_deg))


def load_experiment_from_db(
    db_path: str,
    table: str = "exp_10",
    max_freq: float = 1e8,
) -> pd.DataFrame:
    """
    Expect columns: Freq, Zabs, Phase (deg)
    Only keep (0, max_freq].
    """
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT Freq, Zabs, Phase FROM {table}"
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    df = df.dropna().copy()
    df = df.sort_values("Freq")
    df = df[df["Freq"] > 0]
    df = df[df["Freq"] <= max_freq]
    return df


def sample_freq_points(
    f_all: np.ndarray,
    zabs_all: np.ndarray,
    phase_all: np.ndarray,
    n_samples: int = 800,
    mode: str = "log_uniform",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick subset of experimental points to scan.
    mode:
      - "log_uniform": indices approximately uniform in log(f)
      - "random": random choice in index space
    """
    f_all = np.asarray(f_all)
    N = f_all.size
    if n_samples >= N:
        return f_all, zabs_all, phase_all

    if mode == "log_uniform":
        logf = np.log10(f_all)
        grid = np.linspace(logf.min(), logf.max(), n_samples)
        idx = np.searchsorted(logf, grid)
        idx = np.clip(idx, 0, N - 1)
        idx = np.unique(idx)
        if idx.size < n_samples:
            rng = np.random.default_rng(seed)
            extra = rng.choice(
                np.setdiff1d(np.arange(N), idx),
                size=min(n_samples - idx.size, N - idx.size),
                replace=False,
            )
            idx = np.sort(np.concatenate([idx, extra]))
        return f_all[idx], zabs_all[idx], phase_all[idx]

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_samples, replace=False)
        idx = np.sort(idx)
        return f_all[idx], zabs_all[idx], phase_all[idx]

    raise ValueError(f"Unknown sampling mode: {mode}")


@dataclass
class Params:
    Lls: float
    Csw: float
    Rsw: float
    Llr: float
    Rrs: float
    Rcore: float
    Lm: float
    nLls: float
    Csf: float
    Rsf: float
    Csf0: float
    Lad: float
    Cad: float
    Cad_lad: float   # F, self-capacitance across Lad terminals
    Rad_lad: float   # Ohm, damping in series with Cad_lad branch


def par(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel of two impedances (vectorized)."""
    return 1.0 / (1.0 / a + 1.0 / b)


def par3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Parallel of three impedances (vectorized)."""
    return 1.0 / (1.0 / a + 1.0 / b + 1.0 / c)


def Zmid(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmid = (jωLls) || (1/jωCsw) || Rsw  + Rs"""
    Z_L = 1j * omega * p.Lls
    Z_C = 1.0 / (1j * omega * p.Csw)
    Z_R = p.Rsw + 0j
    Z_par = 1.0 / (1.0 / Z_L + 1.0 / Z_C + 1.0 / Z_R)
    Rs = 8.703
    return Z_par + Rs


def Zmr(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmr = (jωLlr + Rrs) || Rcore || (jωLm)"""
    Z_series = 1j * omega * p.Llr + p.Rrs
    Z_core = p.Rcore + 0j
    Z_Lm = 1j * omega * p.Lm
    return 1.0 / (1.0 / Z_series + 1.0 / Z_core + 1.0 / Z_Lm)


def Zmin(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zmin = Zmid + Zmr"""
    return Zmid(omega, p) + Zmr(omega, p)


def Z_nLls(omega: np.ndarray, p: Params) -> np.ndarray:
    """Z_nLls = jω nLls"""
    return 1j * omega * p.nLls


def Zbra(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zbra = 1/(jωCsf) + Rsf  (Csf series Rsf)"""
    return 1.0 / (1j * omega * p.Csf) + p.Rsf


def Zcsf0(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zcsf0 = 1/(jωCsf0)"""
    return 1.0 / (1j * omega * p.Csf0)


def Zlad(omega: np.ndarray, p: Params) -> np.ndarray:
    """Zlad = jωLad || (Rad_lad + 1/(jωCad_lad))"""
    ZL = 1j * omega * p.Lad
    ZC_branch = p.Rad_lad + 1.0 / (1j * omega * p.Cad_lad)
    return par(ZL, ZC_branch)

def Y_to_Delta(
    Za: np.ndarray,
    Zb: np.ndarray,
    Zc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Y -> Δ, vectorized.
    Returns edges Z1(a-b), Z2(a-c), Z3(b-c)
    """
    S = Za * Zb + Zb * Zc + Zc * Za
    Z1 = S / Zc
    Z2 = S / Zb
    Z3 = S / Za
    return Z1, Z2, Z3


def delta_to_Y(
    Zab: np.ndarray,
    Zbc: np.ndarray,
    Zca: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Δ -> Y, vectorized.
    Returns star arms Za(node a), Zb(node b), Zc(node c)
    """
    S = Zab + Zbc + Zca
    Za = Zab * Zca / S
    Zb = Zab * Zbc / S
    Zc = Zbc * Zca / S
    return Za, Zb, Zc


def Z1_to_Z9(omega: np.ndarray, p: Params):
    Za = Z_nLls(omega, p)
    Zb = Zmin(omega, p)
    Zc = Zbra(omega, p)

    Z1, Z2, Z3 = Y_to_Delta(Za, Zb, Zc)
    Z4_0 = par3(Z3, 0.5 * Z3, Zcsf0(omega, p))

    Za1, Zb1, Zc1 = delta_to_Y(Zab=Z2, Zbc=Z4_0, Zca=Z1)
    Z4, Z5, Z6 = Za1, Zb1, Zc1
    Z7 = Z8 = Z9 = None
    return Z1, Z2, Z3, Z4_0, Z4, Z5, Z6, Z7, Z8, Z9


def Z_total(omega: np.ndarray, p: Params) -> np.ndarray:
    Z1, Z2, _, _, Z4, Z5, Z6, _, _, _ = Z1_to_Z9(omega, p)
    Z_parallel = par(Z6 + 0.5 * Z1, Z5 + 0.5 * Z2)
    Z_core_total = Z_parallel + Z4
    Z_meas = Zlad(omega, p) + Z_core_total + 0.5 * Zlad(omega, p)
    return Z_meas
    # return Z_core_total


def make_initial_params() -> Params:
    return Params(
        Lls=0.0328823,
        Csw=6.33915e-10,
        Rsw=14312.5,
        Llr=0.0138027,
        Rrs=2800.0,
        Rcore=5262.95,
        Lm=0.0206865,
        nLls=3.06593e-10,
        Csf=3.46142e-10,
        Rsf=27.4,
        Csf0=1.98906e-09,
        Lad=1.29349e-07,
        Cad=1e-11,
        Cad_lad=1e-10,
        Rad_lad=47.0,
    )


def clone_params_with_Cad(p: Params, Cad_value: float) -> Params:
    p_new = copy.deepcopy(p)
    p_new.Cad_lad = Cad_value
    return p_new


def _format_cad_title(Cad: float) -> str:
    if Cad < 1e-12:
        return f"Cad = {Cad:.2e} F ({Cad * 1e15:.3g} fF)"
    if Cad < 1e-9:
        return f"Cad = {Cad:.2e} F ({Cad * 1e12:.3g} pF)"
    if Cad < 1e-6:
        return f"Cad = {Cad:.2e} F ({Cad * 1e9:.3g} nF)"
    return f"Cad = {Cad:.2e} F ({Cad * 1e6:.3g} uF)"


def scan_Cad(
    freq_hz: np.ndarray,
    Z_exp: np.ndarray,
    p_base: Params,
    Cad_min: float = 1e-12,
    Cad_max: float = 1e-10,
    n_scan: int = 9,
    n_cols: int = 3,
    phase_ylim: Optional[Tuple[float, float]] = (-100.0, 100.0),
) -> np.ndarray:
    """
    Scan Cad with all other params fixed, then plot |Z| and phase.
    """
    omega = 2 * np.pi * freq_hz
    Cad_list = np.logspace(np.log10(Cad_min), np.log10(Cad_max), n_scan)

    mag_exp = np.abs(Z_exp)
    pha_exp = wrap_phase_deg(np.angle(Z_exp, deg=True))

    n_rows = int(np.ceil(n_scan / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, Cad in enumerate(Cad_list):
        ax = axes[i]
        ax2 = ax.twinx()

        p_test = clone_params_with_Cad(p_base, Cad)
        # debug prints before computing Z_sim
        print("Cad_lad =", p_test.Cad_lad, "Rad_lad =", p_test.Rad_lad)
        Zlad_test = Zlad(omega, p_test)
        idx1 = np.argmin(np.abs(freq_hz - 1e5))
        idx2 = np.argmin(np.abs(freq_hz - 3e7))
        print("Zlad @ 1e5Hz =", Zlad_test[idx1])
        print("Zlad @ 3e7Hz =", Zlad_test[idx2])

        p_test = clone_params_with_Cad(p_base, Cad)
        Z_sim = Z_total(omega, p_test)
        mag_sim = np.abs(Z_sim)
        pha_sim = wrap_phase_deg(np.angle(Z_sim, deg=True))
        # debug prints after computing Z_sim
        Zlad_test_after = Zlad(omega, p_test)
        print("(after) Cad_lad =", p_test.Cad_lad, "Rad_lad =", p_test.Rad_lad)
        print("(after) Zlad @ 1e5Hz =", Zlad_test_after[idx1])
        print("(after) Zlad @ 3e7Hz =", Zlad_test_after[idx2])

        ax.plot(freq_hz, mag_exp, lw=1.2, label="Exp |Z|")
        ax.plot(freq_hz, mag_sim, "--", lw=1.2, label="Sim |Z|")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.5)

        ax2.plot(freq_hz, pha_exp, lw=1.0, alpha=0.75, label="Exp Phase")
        ax2.plot(freq_hz, pha_sim, "--", lw=1.0, alpha=0.75, label="Sim Phase")

        ax.set_title(_format_cad_title(Cad), fontsize=10)

        if i % n_cols == 0:
            ax.set_ylabel("|Z| (Ohm)")
        if i % n_cols == n_cols - 1:
            ax2.set_ylabel("Phase (deg)")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Frequency (Hz)")

        if phase_ylim is not None:
            ax2.set_ylim(*phase_ylim)

        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    for j in range(n_scan, axes.size):
        axes[j].set_visible(False)

    fig.suptitle("Cad scan (A'-BC' shunt capacitor)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

    return Cad_list


def run_scan_Cad_from_db(
    db_path: str,
    table: str,
    p_base: Params,
    max_freq: float = 1e8,
    n_samples: int = 2000,
    sample_mode: str = "log_uniform",
    seed: int = 0,
    Cad_min: float = 1e-12,
    Cad_max: float = 1e-10,
    n_scan: int = 9,
    n_cols: int = 3,
    phase_ylim: Optional[Tuple[float, float]] = (-100.0, 100.0),
) -> np.ndarray:
    """
    Convenience wrapper that mirrors CurVer-style data loading/sampling.
    """
    exp = load_experiment_from_db(db_path, table, max_freq=max_freq)
    f_all = exp["Freq"].to_numpy(dtype=float)
    zabs_all = exp["Zabs"].to_numpy(dtype=float)
    phase_all = exp["Phase"].to_numpy(dtype=float)

    f_scan, zabs_scan, phase_scan = sample_freq_points(
        f_all, zabs_all, phase_all, n_samples=n_samples, mode=sample_mode, seed=seed
    )
    Z_scan = mag_phase_to_complex(zabs_scan, phase_scan)

    return scan_Cad(
        freq_hz=f_scan,
        Z_exp=Z_scan,
        p_base=p_base,
        Cad_min=Cad_min,
        Cad_max=Cad_max,
        n_scan=n_scan,
        n_cols=n_cols,
        phase_ylim=phase_ylim,
    )


def main():
    # ---- user config ----
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"
    MAX_FREQ = 1e8
    N_SAMPLES = 2000
    SAMPLE_MODE = "log_uniform"
    SEED = 0

    # Cad_lad scan range
    CAD_MIN = 1e-12
    CAD_MAX = 1e-10
    N_SCAN = 9
    N_COLS = 3
    PHASE_YLIM = (-100.0, 100.0)
    p0 = make_initial_params()
    run_scan_Cad_from_db(
        db_path=DB_PATH,
        table=TABLE,
        p_base=p0,
        max_freq=MAX_FREQ,
        n_samples=N_SAMPLES,
        sample_mode=SAMPLE_MODE,
        seed=SEED,
        Cad_min=CAD_MIN,
        Cad_max=CAD_MAX,
        n_scan=N_SCAN,
        n_cols=N_COLS,
        phase_ylim=PHASE_YLIM,
    )


if __name__ == "__main__":
    main()
