from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from tmp import (
    Params,
    Z_total,
    load_experiment_from_db,
    mag_phase_to_complex,
    wrap_phase_deg,
)


FIXED_PARAMS = {
    "Lls": 0.0328821,
    "Csw": 6.33885e-10,
    "Rsw": 14312.3,
    "Llr": 0.0138029,
    "Rrs": 2800.0,
    "Rcore": 5263.79,
    "Lm": 0.0206846,
    "nLls": 1.62091e-09,
    "Csf": 3.46171e-10,
    "Rsf": 27.4,
    "Csf0": 6.4003e-10,
    "Rad": 0.01,
}


def _log_space(lo: float, hi: float, n: int) -> np.ndarray:
    if lo <= 0 or hi <= 0:
        raise ValueError("log space requires positive bounds")
    return np.logspace(np.log10(lo), np.log10(hi), n)


def scan_lad_cad(
    db_path: str,
    table: str = "exp_10",
    cad_range: Tuple[float, float] = (3e-13, 3e-12),
    lad_range: Tuple[float, float] = (1e-7, 7e-7),
    n: int = 4,
    n_plot: int = 2000,
    show_exp: bool = True,
) -> None:
    exp = load_experiment_from_db(db_path, table)
    f_all = exp["Freq"].to_numpy(dtype=float)
    zabs_all = exp["Zabs"].to_numpy(dtype=float)
    phase_all = exp["Phase"].to_numpy(dtype=float)
    Z_all = mag_phase_to_complex(zabs_all, phase_all)

    f_plot = np.logspace(np.log10(f_all.min()), np.log10(f_all.max()), n_plot)

    cad_vals = _log_space(cad_range[0], cad_range[1], n)
    lad_vals = _log_space(lad_range[0], lad_range[1], n)

    base = Params(
        Lls=FIXED_PARAMS["Lls"],
        Csw=FIXED_PARAMS["Csw"],
        Rsw=FIXED_PARAMS["Rsw"],
        Llr=FIXED_PARAMS["Llr"],
        Rrs=FIXED_PARAMS["Rrs"],
        Rcore=FIXED_PARAMS["Rcore"],
        Lm=FIXED_PARAMS["Lm"],
        nLls=FIXED_PARAMS["nLls"],
        Csf=FIXED_PARAMS["Csf"],
        Rsf=FIXED_PARAMS["Rsf"],
        Csf0=FIXED_PARAMS["Csf0"],
        Lad=lad_vals[0],
        Cad=cad_vals[0],
        Rad=0.1,
    )

    fig_mag, axes_mag = plt.subplots(n, n, figsize=(4 * n, 3 * n), sharex=True, sharey=True)
    fig_ph, axes_ph = plt.subplots(n, n, figsize=(4 * n, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes_mag = np.array([[axes_mag]])
        axes_ph = np.array([[axes_ph]])

    for i, lad in enumerate(lad_vals):
        for j, cad in enumerate(cad_vals):
            p = replace(base, Lad=float(lad), Cad=float(cad))
            Z_sim = Z_total(2.0 * np.pi * f_plot, p)
            sim_mag = np.log10(np.abs(Z_sim))
            sim_phase = wrap_phase_deg(np.angle(Z_sim, deg=True))

            ax_mag = axes_mag[i, j]
            ax_mag.semilogx(f_plot, sim_mag, color="tab:blue", linewidth=1.5)
            if show_exp:
                ax_mag.semilogx(f_all, np.log10(np.abs(Z_all)), color="tab:orange", alpha=0.35, linewidth=1.0)
            ax_mag.set_title(f"Lad={lad:.2g}, Cad={cad:.2g}")
            ax_mag.grid(True, alpha=0.3)

            ax_ph = axes_ph[i, j]
            ax_ph.semilogx(f_plot, sim_phase, color="tab:blue", linewidth=1.5)
            if show_exp:
                ax_ph.semilogx(f_all, wrap_phase_deg(phase_all), color="tab:orange", alpha=0.35, linewidth=1.0)
            ax_ph.set_title(f"Lad={lad:.2g}, Cad={cad:.2g}")
            ax_ph.grid(True, alpha=0.3)

    fig_mag.suptitle("Lad/Cad Scan (log|Z|)", y=0.995)
    fig_mag.text(0.5, 0.04, "Frequency (Hz)", ha="center")
    fig_mag.text(0.02, 0.5, "log10(|Z|) (Ohm)", va="center", rotation="vertical")
    fig_mag.tight_layout(rect=(0.04, 0.05, 1.0, 0.97))

    fig_ph.suptitle("Lad/Cad Scan (Phase)", y=0.995)
    fig_ph.text(0.5, 0.04, "Frequency (Hz)", ha="center")
    fig_ph.text(0.02, 0.5, "Phase (deg)", va="center", rotation="vertical")
    fig_ph.tight_layout(rect=(0.04, 0.05, 1.0, 0.97))
    plt.show()


if __name__ == "__main__":
    scan_lad_cad(db_path=r"D:\Desktop\EE5003\data\AP_1p5.db", table="exp_10")
