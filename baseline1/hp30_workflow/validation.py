from __future__ import annotations

from pathlib import Path

import pandas as pd

from baseline1 import CurVer
from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex
from baseline1.workflow.model import PARAM_NAMES
from baseline1.workflow.validation import load_params

from .config import Paths
from baseline1.workflow.config import ValidationConfig


def run_validation(paths: Paths, cfg: ValidationConfig, params_path: str | Path) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    params = load_params(params_path)

    print("\n[HP30] Stage 3 validation")
    print(f"[HP30] db = {paths.db_path}")
    print(f"[HP30] table = {cfg.table}")
    print(f"[HP30] params = {params_path}")

    exp = load_experiment(paths.db_path, cfg.table, max_freq=cfg.max_freq)
    f_all, zabs_all, phase_all = frame_to_arrays(exp)
    z_all = mag_phase_to_complex(zabs_all, phase_all)
    z_sim = CurVer.simulate_complex(f_all, params)
    metrics = CurVer.evaluate_raw_space_metrics(z_sim, z_all, len(PARAM_NAMES))
    metrics.update({"table": cfg.table, "params_path": str(params_path)})

    print("[HP30] validation metrics:")
    for key in ("SSE_raw", "RMSE_raw", "AIC_raw", "BIC_raw", "n", "p"):
        print(f"  {key} = {metrics[key]}")

    metrics_path = paths.output_dir / f"stage3_validation_metrics_{cfg.table}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    outputs = {"metrics": metrics_path}

    if cfg.run_gp:
        import matplotlib.pyplot as plt

        plt.show = lambda *args, **kwargs: None
        gp_prefix = paths.output_dir / f"stage3_gp_residual_{cfg.table}"
        gp_csv = paths.output_dir / f"stage3_gp_residual_{cfg.table}.csv"
        CurVer.gp_residual_analysis(
            f_all,
            z_all,
            params,
            out_prefix=str(gp_prefix),
            csv_path=str(gp_csv),
        )
        outputs["gp_png"] = gp_prefix.with_suffix(".png")
        outputs["gp_csv"] = gp_csv
        print(f"[HP30] wrote gp_png: {outputs['gp_png']}")
        print(f"[HP30] wrote gp_csv: {outputs['gp_csv']}")

    print(f"[HP30] wrote metrics: {metrics_path}")
    return outputs
