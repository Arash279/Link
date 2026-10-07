from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from baseline1.workflow.config import ExtractionConfig, FitConfig, InitialGuessConfig, ValidationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HP30_DIR = PROJECT_ROOT / "baseline1" / "hp30_workflow"


def default_db_path() -> Path:
    candidates = (
        Path(r"D:\Desktop\EE5003\data\AP_30.db"),
        Path(r"D:\Desktop\paper\EE5003\data\AP_30.db"),
        Path(r"D:\Desktop\Data\AP_30.db"),
        Path(r"D:\Desktop\data\AP_30.db"),
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


@dataclass(frozen=True)
class Paths:
    db_path: Path = field(default_factory=default_db_path)
    output_dir: Path = HP30_DIR / "outputs"


@dataclass(frozen=True)
class WorkflowConfig:
    paths: Paths = field(default_factory=Paths)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    initial_guess: InitialGuessConfig = field(default_factory=InitialGuessConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
