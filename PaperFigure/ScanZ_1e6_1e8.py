# -*- coding: utf-8 -*-
"""
Plot fixed-parameter impedance against experiment data from SQLite.
GP residual analysis is intentionally disabled in this script.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FREQ_PLOT_RANGE = (1e6, 1e8)


def par(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Parallel of two impedances (vectorized)."""
    return a * b / (a + b)


def par3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Parallel of three impedances (vectorized)."""
    return 1.0 / (1.0 / a + 1.0 / b + 1.0 / c)


def wrap_phase_deg(phi_deg: np.ndarray) -> np.ndarray:
    """Wrap phase to (-180, 180]."""
    return (phi_deg + 180.0) % 360.0 - 180.0


def phase_diff_deg(phi_sim: np.ndarray, phi_exp: np.ndarray) -> np.ndarray:
    """Smallest signed difference between two phases (deg)."""
    d = phi_sim - phi_exp
    return (d + 180.0) % 360.0 - 180.0


def mag_phase_to_complex(mag: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Convert magnitude + phase(deg) to complex."""
    return mag * np.exp(1j * np.deg2rad(phase_deg))


PARAM_NAMES: List[str] = [
    "Lls",
    "Csw",
    "Rsw",
    "Llr",
    "Rrs",
    "Rcore",
    "Lm",
    "nLls",
    "Csf",
    "Rsf",
    "Csf0",
]
N_PARAMS: int = len(PARAM_NAMES)


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

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "Params":
        missing = [k for k in PARAM_NAMES if k not in d]
        extra = [k for k in d.keys() if k not in PARAM_NAMES]
        if missing:
            raise ValueError(f"Missing params: {missing}")
        if extra:
            raise ValueError(f"Unknown params: {extra}")
        return Params(**{k: float(d[k]) for k in PARAM_NAMES})

    def as_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in PARAM_NAMES}


Rs = 8.703


def Zmid(omega: np.ndarray, p: Params) -> np.ndarray:
    z_l = 1j * omega * p.Lls
    z_c = 1.0 / (1j * omega * p.Csw)
    z_r = p.Rsw + 0j
    z_par = 1.0 / (1.0 / z_l + 1.0 / z_c + 1.0 / z_r)
    return z_par + Rs


def Zmr(omega: np.ndarray, p: Params) -> np.ndarray:
    z_series = 1j * omega * p.Llr + p.Rrs
    z_core = p.Rcore + 0j
    z_lm = 1j * omega * p.Lm
    return 1.0 / (1.0 / z_series + 1.0 / z_core + 1.0 / z_lm)


def Zmin(omega: np.ndarray, p: Params) -> np.ndarray:
    return Zmid(omega, p) + Zmr(omega, p)


def Z_nLls(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1j * omega * p.nLls


def Zbra(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1.0 / (1j * omega * p.Csf) + p.Rsf


def Zcsf0(omega: np.ndarray, p: Params) -> np.ndarray:
    return 1.0 / (1j * omega * p.Csf0)


def Y_to_Delta(
    za: np.ndarray, zb: np.ndarray, zc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = za * zb + zb * zc + zc * za
    z1 = s / zc
    z2 = s / zb
    z3 = s / za
    return z1, z2, z3


def delta_to_Y(
    zab: np.ndarray, zbc: np.ndarray, zca: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = zab + zbc + zca
    za = zab * zca / s
    zb = zab * zbc / s
    zc = zbc * zca / s
    return za, zb, zc


def Z1_to_Z9(omega: np.ndarray, p: Params):
    za = Z_nLls(omega, p)
    zb = Zmin(omega, p)
    zc = Zbra(omega, p)

    z1, z2, z3 = Y_to_Delta(za, zb, zc)
    z4_0 = par3(z3, 0.5 * z3, Zcsf0(omega, p))

    za1, zb1, zc1 = delta_to_Y(zab=z2, zbc=z4_0, zca=z1)
    z4, z5, z6 = za1, zb1, zc1

    z7 = z8 = z9 = None
    return z1, z2, z3, z4_0, z4, z5, z6, z7, z8, z9


def Z_total(omega: np.ndarray, p: Params) -> np.ndarray:
    z1, z2, _, _, z4, z5, z6, _, _, _ = Z1_to_Z9(omega, p)
    z_parallel = par(z6 + 0.5 * z1, z5 + 0.5 * z2)
    z_core_total = z_parallel + z4
    return z_core_total


def load_experiment_from_db(db_path: str, table: str) -> pd.DataFrame:
    """Expect columns: Freq, Zabs, Phase (deg)."""
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT Freq, Zabs, Phase FROM {table}"
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    df = df.dropna().copy()
    df = df.sort_values("Freq")
    df = df[df["Freq"] > 0]
    return df


def select_frequency_band(
    f: np.ndarray,
    *arrays: np.ndarray,
    f_range: Tuple[float, float],
) -> Tuple[np.ndarray, ...]:
    """Apply the same frequency-band mask to frequency and aligned arrays."""
    f = np.asarray(f, dtype=float)
    f_lo, f_hi = f_range
    mask = (f >= f_lo) & (f <= f_hi)
    out = [f[mask]]
    for arr in arrays:
        out.append(np.asarray(arr)[mask])
    return tuple(out)


def simulate_on_freq(f_hz: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns: Z (complex), log10(|Z|), wrapped phase."""
    f_hz = np.asarray(f_hz, dtype=float)
    omega = 2.0 * np.pi * f_hz
    z = Z_total(omega, p)
    mag = np.abs(z)
    ph = np.angle(z, deg=True)
    return z, np.log10(mag), wrap_phase_deg(ph)


def component_impedances(f_hz: np.ndarray, p: Params) -> Dict[str, np.ndarray]:
    """Return individual component impedances on the given frequency grid."""
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    return {
        "Zmr": Zmr(omega, p),
        "Zmid": Zmid(omega, p),
        "Z_nLls": Z_nLls(omega, p),
        "Zbra": Zbra(omega, p),
        "Zcsf0": Zcsf0(omega, p),
    }


def compute_metrics(
    f: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp: np.ndarray,
    logmag_sim: np.ndarray,
    phase_sim: np.ndarray,
    f_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Error metrics on log10|Z| and wrapped phase."""
    f = np.asarray(f, float)
    zabs_exp = np.asarray(zabs_exp, float)
    phase_exp = wrap_phase_deg(np.asarray(phase_exp, float))

    if f_range is not None:
        f_lo, f_hi = f_range
        mask = (f >= f_lo) & (f <= f_hi)
    else:
        mask = np.ones_like(f, dtype=bool)

    if mask.sum() < 5:
        raise ValueError("Too few points in selected frequency range.")

    logmag_exp = np.log10(zabs_exp[mask])
    logmag_err = logmag_sim[mask] - logmag_exp
    phase_err = phase_diff_deg(phase_sim[mask], phase_exp[mask])

    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    def mae(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))

    return {
        "N_points": int(mask.sum()),
        "logmag_RMSE": rmse(logmag_err),
        "logmag_MAE": mae(logmag_err),
        "phase_RMSE_deg": rmse(phase_err),
        "phase_MAE_deg": mae(phase_err),
    }


def compute_complex_residual(
    Z_sim: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp_deg: np.ndarray,
) -> np.ndarray:
    """Rebuild Z_exp from magnitude/phase and return Z_exp - Z_sim."""
    Z_exp = mag_phase_to_complex(zabs_exp, phase_exp_deg)
    return Z_exp - Z_sim


def plot_compare(
    f_exp: np.ndarray,
    zabs_exp: np.ndarray,
    phase_exp: np.ndarray,
    f_sim: np.ndarray,
    zabs_sim: np.ndarray,
    phase_sim: np.ndarray,
    title_suffix: str = "",
):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.semilogx(f_sim, np.log10(zabs_sim), label="Simulation", linewidth=2)
    plt.semilogx(f_exp, np.log10(zabs_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(|Z|) (Ohm)")
    plt.title(f"Impedance Magnitude Comparison {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogx(f_sim, wrap_phase_deg(phase_sim), label="Simulation", linewidth=2)
    plt.semilogx(f_exp, wrap_phase_deg(phase_exp), label="Experiment", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Impedance Phase Comparison {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_component_impedances(
    f_hz: np.ndarray,
    components: Dict[str, np.ndarray],
    title_suffix: str = "",
):
    """Plot absolute magnitude and phase of individual impedances."""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for name, z in components.items():
        plt.semilogx(f_hz, np.abs(z), label=name, linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|Z| (Ohm)")
    plt.title(f"Component Impedance Magnitude {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for name, z in components.items():
        plt.semilogx(f_hz, wrap_phase_deg(np.angle(z, deg=True)), label=name, linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Component Impedance Phase {title_suffix}".strip())
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_component_table(
    f_hz: np.ndarray,
    components: Dict[str, np.ndarray],
    n_points: int = 6,
):
    """Print a compact table at a few log-spaced frequency points."""
    f_hz = np.asarray(f_hz, dtype=float)
    if f_hz.size == 0:
        return

    target_freqs = np.geomspace(f_hz[0], f_hz[-1], num=min(n_points, f_hz.size))
    sample_idx = np.unique([int(np.argmin(np.abs(f_hz - f0))) for f0 in target_freqs])

    print("\n===== Component impedance samples =====")
    for idx in sample_idx:
        print(f"\nFreq = {f_hz[idx]:.6g} Hz")
        print(f"{'Name':8s} {'|Z|(Ohm)':>14s} {'Phase(deg)':>14s}")
        for name, z in components.items():
            mag = np.abs(z[idx])
            phase = wrap_phase_deg(np.array([np.angle(z[idx], deg=True)]))[0]
            print(f"{name:8s} {mag:14.6g} {phase:14.6f}")


def plot_residuals(
    f: np.ndarray,
    res: np.ndarray,
    title: str,
    rel_to: np.ndarray | None = None,
):
    """Plot residual Re, Im and |res|."""
    f = np.asarray(f, float)
    re = np.real(res)
    im = np.imag(res)
    mag = np.abs(res)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title)

    axes[0].semilogx(f, re, ".", color="tab:blue", alpha=0.4, label="Residual Re")
    axes[0].set_ylabel("Re residual (Ohm)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].semilogx(f, im, ".", color="tab:blue", alpha=0.4, label="Residual Im")
    axes[1].set_ylabel("Im residual (Ohm)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].semilogx(f, mag, ".", color="tab:blue", alpha=0.4, label="|res|")
    if rel_to is not None:
        rel = mag / (np.asarray(rel_to, float) + 1e-12)
        axes[2].semilogx(f, rel, "r-", linewidth=1.5, label="|res| / |Z_exp|")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude (Ohm/ratio)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
    TABLE = "exp_10"

    PARAMS_FIXED: Dict[str, float] = dict(
        Lls=0.03288,
        Csw=6.33796e-10,
        Rsw=14311.4,
        Llr=0.0138097,
        Rrs=2800,
        Rcore=5265.38,
        Lm=0.0206859,
        nLls=1.7806e-09,
        Csf=3.46289e-10,
        Rsf=27.4,
        Csf0=7.38e-09,
    )

    METRIC_BANDS = [
        FREQ_PLOT_RANGE,
    ]

    exp = load_experiment_from_db(DB_PATH, TABLE)
    f = exp["Freq"].to_numpy(dtype=float)
    zabs_exp = exp["Zabs"].to_numpy(dtype=float)
    phase_exp = exp["Phase"].to_numpy(dtype=float)

    p = Params.from_dict(PARAMS_FIXED)
    z, logmag_sim, phase_sim = simulate_on_freq(f, p)
    zabs_sim = np.abs(z)
    components = component_impedances(f, p)
    f_plot, zabs_exp_plot, phase_exp_plot, z_plot, zabs_sim_plot, phase_sim_plot = select_frequency_band(
        f,
        zabs_exp,
        phase_exp,
        z,
        zabs_sim,
        phase_sim,
        f_range=FREQ_PLOT_RANGE,
    )
    components_plot = {name: values[(f >= FREQ_PLOT_RANGE[0]) & (f <= FREQ_PLOT_RANGE[1])] for name, values in components.items()}

    print("\n===== Fixed parameters used (NO FIX) =====")
    for k, v in p.as_dict().items():
        print(f"{k:8s} = {v:.6g}")

    print("\n===== Error metrics =====")
    for band in METRIC_BANDS:
        tag = "Full band" if band is None else f"{band[0]:.3g} ~ {band[1]:.3g} Hz"
        m = compute_metrics(
            f=f,
            zabs_exp=zabs_exp,
            phase_exp=phase_exp,
            logmag_sim=logmag_sim,
            phase_sim=phase_sim,
            f_range=band,
        )
        print(f"\n[{tag}]  N={m['N_points']}")
        print(f"  log10|Z| RMSE = {m['logmag_RMSE']:.6g}")
        print(f"  log10|Z| MAE  = {m['logmag_MAE']:.6g}")
        print(f"  Phase RMSE(deg)= {m['phase_RMSE_deg']:.6g}")
        print(f"  Phase MAE(deg) = {m['phase_MAE_deg']:.6g}")

    print_component_table(f_hz=f_plot, components=components_plot)

    plot_compare(
        f_exp=f_plot,
        zabs_exp=zabs_exp_plot,
        phase_exp=phase_exp_plot,
        f_sim=f_plot,
        zabs_sim=zabs_sim_plot,
        phase_sim=phase_sim_plot,
        title_suffix="(1e6 ~ 1e8 Hz, NO FIX)",
    )
    plot_component_impedances(
        f_hz=f_plot,
        components=components_plot,
        title_suffix="(1e6 ~ 1e8 Hz, NO FIX)",
    )

    # GP residual analysis is disabled.
    # Z_exp = mag_phase_to_complex(zabs_exp, phase_exp)
    # gp_residual_analysis(
    #     f_hz=f,
    #     Z_exp=Z_exp,
    #     Z_sim=z,
    #     out_prefix=None,
    #     top_n=3,
    # )


if __name__ == "__main__":
    main()
