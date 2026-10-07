from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from baseline1 import CurVer
from baseline1.workflow.config import ExtractionConfig, FitConfig, InitialGuessConfig, Paths
from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex, sample_freq_points
from baseline1.workflow.model import PARAM_NAMES, Params, simulate_complex, simulate_on_freq
from baseline1.workflow.parameter_fitting import run_parameter_fitting_stage, write_parameter_fitting_outputs
from baseline1.workflow.initial_params import build_initial_param_values


def _params_to_records(params: Params) -> list[dict[str, float | str]]:
    return [{"parameter": name, "value": float(getattr(params, name))} for name in PARAM_NAMES]


def phase_diff_deg(phi_sim: np.ndarray, phi_exp: np.ndarray) -> np.ndarray:
    return (phi_sim - phi_exp + 180.0) % 360.0 - 180.0


def make_mag_phase_residual_fn(
    f_hz: np.ndarray,
    logmag_data: np.ndarray,
    phase_data: np.ndarray,
    weights: np.ndarray,
    w_logmag: float,
    w_phase: float,
):
    f_hz = np.asarray(f_hz, dtype=float)
    logmag_data = np.asarray(logmag_data, dtype=float)
    phase_data = np.asarray(phase_data, dtype=float)
    weights = np.asarray(weights, dtype=float)

    def residual(u: np.ndarray) -> np.ndarray:
        params = Params.from_vector(np.exp(u))
        logmag_sim, phase_sim = simulate_on_freq(f_hz, params)
        r_mag = weights * (logmag_sim - logmag_data) * w_logmag
        r_phase = weights * phase_diff_deg(phase_sim, phase_data) * w_phase
        return np.concatenate([r_mag, r_phase])

    return residual


def fit_params_de_ls_mag_phase(
    f_fit: np.ndarray,
    zabs_fit: np.ndarray,
    phase_fit: np.ndarray,
    p0: Params,
    weights: np.ndarray,
    cfg: FitConfig,
    w_logmag: float = 1.0,
    w_phase: float = 1.0 / 30.0,
) -> Tuple[Params, list[dict[str, Any]]]:
    from scipy.optimize import differential_evolution, least_squares

    lo, hi = CurVer.default_bounds(p0)
    u_lo = np.log(lo)
    u_hi = np.log(hi)
    logmag_fit = np.log10(zabs_fit)
    residual = make_mag_phase_residual_fn(f_fit, logmag_fit, phase_fit, weights, w_logmag, w_phase)
    residual0 = residual(np.log(p0.to_vector()))
    f_scale = max(CurVer.mad(residual0), 1e-6)

    def objective(u: np.ndarray) -> float:
        r = residual(u)
        return 0.5 * float(np.dot(r, r))

    de_res = differential_evolution(
        objective,
        bounds=list(zip(u_lo.tolist(), u_hi.tolist())),
        maxiter=cfg.de_maxiter,
        popsize=cfg.de_popsize,
        polish=False,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed + 1)
    starts = rng.uniform(u_lo, u_hi, size=(cfg.n_starts, len(PARAM_NAMES)))
    scores = np.array([objective(u) for u in starts], dtype=float)
    top_idx = np.argsort(scores)[: max(cfg.top_k, 1)]
    candidates = [de_res.x, *[starts[i] for i in top_idx]]

    results: list[dict[str, Any]] = []
    for start in candidates:
        res = least_squares(
            residual,
            start,
            bounds=(u_lo, u_hi),
            max_nfev=cfg.max_nfev,
            loss=cfg.loss,
            f_scale=f_scale,
            verbose=0,
        )
        results.append(
            {
                "x": res.x,
                "cost": float(res.cost),
                "status": int(res.status),
                "message": str(res.message),
                "nfev": int(res.nfev),
            }
        )

    results.sort(key=lambda item: item["cost"])
    return Params.from_vector(np.exp(results[0]["x"])), results


def run_fit(
    paths: Paths,
    cfg: FitConfig,
    extraction_cfg: ExtractionConfig,
    priors: InitialGuessConfig,
) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_results = run_parameter_fitting_stage(paths, extraction_cfg, priors)
    write_parameter_fitting_outputs(paths, parameter_results)

    exp = load_experiment(paths.db_path, cfg.table, max_freq=cfg.max_freq)
    f_all, zabs_all, phase_all = frame_to_arrays(exp)
    f_fit, zabs_fit, phase_fit = sample_freq_points(
        f_all,
        zabs_all,
        phase_all,
        n_samples=cfg.n_samples,
        mode=cfg.sample_mode,
        seed=cfg.seed,
    )
    z_fit = mag_phase_to_complex(zabs_fit, phase_fit)
    p0 = Params(**build_initial_param_values(parameter_results["initial_inputs"]))
    weights = CurVer.compute_freq_weights(
        f_fit,
        z_fit,
        mode=cfg.weight_mode,
        min_w=cfg.weight_min,
        max_w=cfg.weight_max,
        power=cfg.weight_power,
    )

    t0 = time.perf_counter()
    p_opt, results = fit_params_de_ls_mag_phase(f_fit, zabs_fit, phase_fit, p0, weights, cfg)
    elapsed = time.perf_counter() - t0

    z_all = mag_phase_to_complex(zabs_all, phase_all)
    z_sim_all = simulate_complex(f_all, p_opt)
    metrics = CurVer.evaluate_raw_space_metrics(z_sim_all, z_all, len(PARAM_NAMES))
    metrics.update(
        {
            "table": cfg.table,
            "seed": cfg.seed,
            "fit_seconds": elapsed,
            "fit_space": "logmag_phase",
            "best_cost": float(results[0]["cost"]),
            "best_nfev": int(results[0]["nfev"]),
            "n_freq_fit": int(f_fit.size),
        }
    )

    prefix = f"mag_phase_fit_{cfg.table}_seed{cfg.seed}"
    params_path = paths.output_dir / f"{prefix}_params.json"
    params_csv_path = paths.output_dir / f"{prefix}_params.csv"
    metrics_path = paths.output_dir / f"{prefix}_metrics.csv"
    candidates_path = paths.output_dir / f"{prefix}_candidates.csv"

    payload = {
        "table": cfg.table,
        "seed": cfg.seed,
        "fit_space": "logmag_phase",
        "initial_inputs": {
            name: float(value) if isinstance(value, (int, float)) else value
            for name, value in parameter_results["initial_inputs"].__dict__.items()
        },
        "parameters": {name: float(getattr(p_opt, name)) for name in PARAM_NAMES},
        "metrics": metrics,
    }
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(_params_to_records(p_opt)).to_csv(params_csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "rank": rank,
                "cost": float(item["cost"]),
                "nfev": int(item["nfev"]),
                "status": int(item["status"]),
                "message": item["message"],
            }
            for rank, item in enumerate(results, start=1)
        ]
    ).to_csv(candidates_path, index=False, encoding="utf-8-sig")

    return {
        "params_json": params_path,
        "params_csv": params_csv_path,
        "metrics": metrics_path,
        "candidates": candidates_path,
    }
