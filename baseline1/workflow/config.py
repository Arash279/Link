from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "baseline1"


def default_db_path() -> Path:
    candidates = (
        Path(r"D:\Desktop\EE5003\data\AP_1p5.db"),
        Path(r"D:\Desktop\paper\EE5003\data\AP_1p5.db"),
        Path(r"D:\Desktop\Data\AP_1p5.db"),
        Path(r"D:\Desktop\data\AP_1p5.db"),
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


@dataclass(frozen=True)
class Paths:
    db_path: Path = field(default_factory=default_db_path)
    output_dir: Path = BASELINE_DIR / "workflow_outputs"


@dataclass(frozen=True)
class ExtractionConfig:
    y_short_tables: Sequence[str] = field(default_factory=lambda: ("exp_10", "exp_11", "exp_12"))
    y_long_tables: Sequence[str] = field(default_factory=lambda: ("exp_14", "exp_15", "exp_16"))
    delta_tables: Sequence[str] = field(default_factory=lambda: ("exp_18", "exp_19", "exp_20"))
    single_phase_tables: Sequence[str] = field(default_factory=lambda: tuple(f"exp_{i}" for i in range(1, 7)))
    rr_tables: Sequence[str] = field(default_factory=lambda: ("exp_10", "exp_11", "exp_12"))
    lm_primary_tables: Sequence[str] = field(default_factory=lambda: ("exp_10", "exp_11", "exp_12"))
    lm_check_tables: Sequence[str] = field(default_factory=lambda: ("exp_18", "exp_19", "exp_20"))
    rr_fmin_hz: float = 100.0
    rr_fmax_hz: float = 500.0


@dataclass(frozen=True)
class InitialGuessConfig:
    hp: float = 1.5
    connection: Literal["Y", "Delta"] = "Y"
    nlls_init_h: float = 1.8e-10
    lad_h: float = 1.3e-7


@dataclass(frozen=True)
class FitConfig:
    table: str = "exp_10"
    max_freq: float = 1e8
    n_samples: int = 2000
    sample_mode: str = "log_uniform"
    seed: int = 0
    n_starts: int = 120
    top_k: int = 10
    max_nfev: int = 200
    de_maxiter: int = 60
    de_popsize: int = 10
    weight_mode: str = "auto"
    weight_min: float = 0.3
    weight_max: float = 4.0
    weight_power: float = 1.0
    scale_mode: str = "mad"
    loss: str = "soft_l1"


@dataclass(frozen=True)
class ValidationConfig:
    table: str = "exp_10"
    max_freq: float = 1e8
    run_gp: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    paths: Paths = field(default_factory=Paths)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    initial_guess: InitialGuessConfig = field(default_factory=InitialGuessConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
