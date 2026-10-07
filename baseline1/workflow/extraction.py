from __future__ import annotations

from pathlib import Path

from .config import ExtractionConfig, InitialGuessConfig, Paths
from .parameter_fitting import run_parameter_fitting_stage, write_parameter_fitting_outputs


def run_extraction(
    paths: Paths,
    cfg: ExtractionConfig,
    priors: InitialGuessConfig,
) -> dict[str, Path]:
    results = run_parameter_fitting_stage(paths, cfg, priors)
    return write_parameter_fitting_outputs(paths, results)
