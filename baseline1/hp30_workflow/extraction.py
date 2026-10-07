from __future__ import annotations

from pathlib import Path

from baseline1.workflow.config import ExtractionConfig, InitialGuessConfig
from baseline1.workflow.extraction import run_extraction as run_base_extraction

from .config import Paths


def run_extraction(paths: Paths, cfg: ExtractionConfig, priors: InitialGuessConfig) -> dict[str, Path]:
    return run_base_extraction(paths, cfg, priors)
