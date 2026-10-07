from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline1.workflow.config import WorkflowConfig
from baseline1.workflow.extraction import run_extraction
from baseline1.workflow.validation import run_validation

if __package__ in {None, ""}:
    from baseline1.experimental_reim_workflow.fitting import run_fit
else:
    from .fitting import run_fit


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental Re/Im fitting workflow.")
    parser.add_argument("--stage", choices=("extract", "fit", "validate", "all"), default="all")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table", default="exp_10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params-path", default=None)
    parser.add_argument("--no-gp", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = WorkflowConfig()
    paths = cfg.paths
    if args.db_path:
        paths = replace(paths, db_path=Path(args.db_path))
    if args.output_dir:
        paths = replace(paths, output_dir=Path(args.output_dir))
    else:
        paths = replace(paths, output_dir=DEFAULT_OUTPUT_DIR)

    fit_cfg = replace(cfg.fit, table=args.table, seed=args.seed)
    val_cfg = replace(cfg.validation, table=args.table, run_gp=not args.no_gp)

    latest_params = None
    if args.stage in {"extract", "all"}:
        outputs = run_extraction(paths, cfg.extraction, cfg.initial_guess)
        print("Stage 1 extraction outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.stage in {"fit", "all"}:
        outputs = run_fit(paths, fit_cfg, cfg.extraction, cfg.initial_guess)
        latest_params = outputs["params_json"]
        print("Experimental Re/Im fit outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.stage in {"validate", "all"}:
        params_path = latest_params or args.params_path
        if not params_path:
            raise SystemExit("--params-path is required when running validate directly.")
        outputs = run_validation(paths, val_cfg, params_path)
        print("Stage 3 validation outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
