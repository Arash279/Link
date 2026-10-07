from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline1 import CurVer
from baseline1.workflow.config import ExtractionConfig, FitConfig, InitialGuessConfig
from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex, sample_freq_points
from baseline1.workflow.initial_params import DEFAULT_INITIAL_INPUTS, build_initial_param_values
from baseline1.workflow.model import PARAM_NAMES
from baseline1.workflow.parameter_fitting import run_parameter_fitting_stage, write_parameter_fitting_outputs

from .config import Paths


def _params_to_records(params: CurVer.Params) -> list[dict[str, float | str]]:
    return [{"parameter": name, "value": float(getattr(params, name))} for name in PARAM_NAMES]


def _print_section(title: str) -> None:
    print(f"\n[HP30] {title}")


def _print_frame(name: str, frame: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 12) -> None:
    print(f"[HP30] {name}:")
    if frame.empty:
        print("  <empty>")
        return
    view = frame[cols] if cols else frame
    print(view.head(max_rows).to_string(index=False))


def _print_initial_inputs(inputs: object) -> None:
    values = inputs.__dict__
    print("[HP30] stage-1 initial inputs:")
    for key, value in values.items():
        print(f"  {key} = {value}")


def run_fit(
    paths: Paths,
    cfg: FitConfig,
    extraction_cfg: ExtractionConfig,
    priors: InitialGuessConfig,
) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    _print_section("Stage 1 parameter fitting")
    print(f"[HP30] db = {paths.db_path}")
    parameter_results = run_parameter_fitting_stage(paths, extraction_cfg, priors)
    stage1_outputs = write_parameter_fitting_outputs(paths, parameter_results)
    for name, path in stage1_outputs.items():
        print(f"[HP30] wrote {name}: {path}")

    _print_frame(
        "resonance summary",
        parameter_results["resonance"],
        ["group", "table", "fr_hz", "zmax_ohm", "fa_hz", "zanti_ohm"],
    )
    _print_frame(
        "leakage summary",
        parameter_results["leakage"],
        ["group", "table", "l_sigma_h", "err", "quality_ok"] if "quality_ok" in parameter_results["leakage"] else None,
    )
    _print_frame("Rrs summary", parameter_results["rr"])
    _print_frame("Lm summary", parameter_results["lm"])
    _print_initial_inputs(parameter_results["initial_inputs"])

    _print_section("Stage 2 DE + LS fit")
    exp = load_experiment(paths.db_path, cfg.table, max_freq=cfg.max_freq)
    f_all, zabs_all, phase_all = frame_to_arrays(exp)
    print(f"[HP30] fit table = {cfg.table}")
    print(f"[HP30] total data points = {f_all.size}")
    f_fit, zabs_fit, phase_fit = sample_freq_points(
        f_all,
        zabs_all,
        phase_all,
        n_samples=cfg.n_samples,
        mode=cfg.sample_mode,
        seed=cfg.seed,
    )
    print(f"[HP30] sampled fit points = {f_fit.size} (mode={cfg.sample_mode}, seed={cfg.seed})")

    z_fit = mag_phase_to_complex(zabs_fit, phase_fit)
    fit_inputs = DEFAULT_INITIAL_INPUTS
    print("[HP30] note: stage-1 diagnostics above are for inspection only.")
    print("[HP30] note: fit initial inputs are copied from the main workflow defaults.")
    _print_initial_inputs(fit_inputs)
    p0 = CurVer.Params(**build_initial_param_values(fit_inputs))
    print("[HP30] initial parameter vector used by the optimizer:")
    for name in PARAM_NAMES:
        print(f"  {name} = {getattr(p0, name)}")

    weights = CurVer.compute_freq_weights(
        f_fit,
        z_fit,
        mode=cfg.weight_mode,
        min_w=cfg.weight_min,
        max_w=cfg.weight_max,
        power=cfg.weight_power,
    )
    if cfg.scale_mode == "mad":
        s_re = max(CurVer.mad(z_fit.real), 1e-12)
        s_im = max(CurVer.mad(z_fit.imag), 1e-12)
    elif cfg.scale_mode == "std":
        s_re = max(float(np.std(z_fit.real)), 1e-12)
        s_im = max(float(np.std(z_fit.imag)), 1e-12)
    else:
        raise ValueError(f"Unknown scale_mode: {cfg.scale_mode}")
    print(f"[HP30] residual scaling: s_re={s_re:.6g}, s_im={s_im:.6g}")
    print(f"[HP30] DE config: maxiter={cfg.de_maxiter}, popsize={cfg.de_popsize}, n_starts={cfg.n_starts}, top_k={cfg.top_k}")

    residual0 = CurVer.make_residual_fn(f_fit, z_fit, weights, s_re, s_im)(np.log(p0.to_vector()))
    f_scale = max(CurVer.mad(residual0), 1e-6)
    print(f"[HP30] robust loss = {cfg.loss}, f_scale = {f_scale:.6g}")

    t0 = time.perf_counter()
    p_opt, results = CurVer.fit_params_global_local(
        f_fit=f_fit,
        Z_fit=z_fit,
        p0=p0,
        weights=weights,
        s_re=s_re,
        s_im=s_im,
        n_starts=cfg.n_starts,
        top_k=cfg.top_k,
        seed=cfg.seed,
        global_method="de",
        max_nfev=cfg.max_nfev,
        loss=cfg.loss,
        f_scale=f_scale,
        de_maxiter=cfg.de_maxiter,
        de_popsize=cfg.de_popsize,
    )
    elapsed = time.perf_counter() - t0

    print("[HP30] top candidate summary:")
    for rank, item in enumerate(results[: min(5, len(results))], start=1):
        print(
            f"  rank={rank} cost={float(item['cost']):.6g} nfev={int(item['nfev'])} "
            f"status={item['status']} message={item['message']}"
        )

    print("[HP30] fitted parameter vector:")
    for name in PARAM_NAMES:
        print(f"  {name} = {getattr(p_opt, name)}")

    z_all = mag_phase_to_complex(zabs_all, phase_all)
    z_sim_all = CurVer.simulate_complex(f_all, p_opt)
    metrics = CurVer.evaluate_raw_space_metrics(z_sim_all, z_all, len(PARAM_NAMES))
    metrics.update(
        {
            "table": cfg.table,
            "seed": cfg.seed,
            "fit_seconds": elapsed,
            "best_cost": float(results[0]["cost"]),
            "best_nfev": int(results[0]["nfev"]),
            "n_freq_fit": int(f_fit.size),
        }
    )
    print("[HP30] fit metrics:")
    for key in ("SSE_raw", "RMSE_raw", "AIC_raw", "BIC_raw", "fit_seconds", "best_cost", "best_nfev"):
        print(f"  {key} = {metrics[key]}")

    params_path = paths.output_dir / f"stage2_fit_params_{cfg.table}_seed{cfg.seed}.json"
    params_csv_path = paths.output_dir / f"stage2_fit_params_{cfg.table}_seed{cfg.seed}.csv"
    metrics_path = paths.output_dir / f"stage2_fit_metrics_{cfg.table}_seed{cfg.seed}.csv"
    candidates_path = paths.output_dir / f"stage2_fit_candidates_{cfg.table}_seed{cfg.seed}.csv"

    payload: dict[str, Any] = {
        "table": cfg.table,
        "seed": cfg.seed,
        "initial_inputs": {
            name: float(value) if isinstance(value, (int, float)) else value
            for name, value in fit_inputs.__dict__.items()
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
                "status": item["status"],
                "message": str(item["message"]),
            }
            for rank, item in enumerate(results, start=1)
        ]
    ).to_csv(candidates_path, index=False, encoding="utf-8-sig")

    print(f"[HP30] wrote params_json: {params_path}")
    print(f"[HP30] wrote params_csv: {params_csv_path}")
    print(f"[HP30] wrote metrics: {metrics_path}")
    print(f"[HP30] wrote candidates: {candidates_path}")

    return {
        "params_json": params_path,
        "params_csv": params_csv_path,
        "metrics": metrics_path,
        "candidates": candidates_path,
    }
