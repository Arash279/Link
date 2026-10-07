from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from baseline1.hp30_workflow.config import WorkflowConfig
    from baseline1.hp30_workflow.extraction import run_extraction
    from baseline1.hp30_workflow.fitting import run_fit
    from baseline1.hp30_workflow.validation import run_validation
else:
    from .config import WorkflowConfig
    from .extraction import run_extraction
    from .fitting import run_fit
    from .validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HP_30 workflow with detailed logging.")
    parser.add_argument("--stage", choices=("extract", "fit", "validate", "all"), default="all")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table", default="exp_10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params-path", default=None, help="Required for validate unless running stage=all.")
    parser.add_argument("--no-gp", action="store_true", help="Skip GP residual validation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = WorkflowConfig()
    paths = cfg.paths
    if args.db_path:
        paths = replace(paths, db_path=Path(args.db_path))
    if args.output_dir:
        paths = replace(paths, output_dir=Path(args.output_dir))

    print("[HP30] workflow bootstrap")
    print(f"[HP30] db_path = {paths.db_path}")
    print(f"[HP30] output_dir = {paths.output_dir}")
    print(f"[HP30] stage = {args.stage}, table = {args.table}, seed = {args.seed}")

    fit_cfg = replace(cfg.fit, table=args.table, seed=args.seed)
    val_cfg = replace(cfg.validation, table=args.table, run_gp=not args.no_gp)

    latest_params = None
    if args.stage in {"extract", "all"}:
        outputs = run_extraction(paths, cfg.extraction, cfg.initial_guess)
        print("\n[HP30] Stage 1 extraction outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.stage in {"fit", "all"}:
        outputs = run_fit(paths, fit_cfg, cfg.extraction, cfg.initial_guess)
        latest_params = outputs["params_json"]
        print("\n[HP30] Stage 2 fit outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if args.stage in {"validate", "all"}:
        params_path = latest_params or args.params_path
        if not params_path:
            raise SystemExit("--params-path is required when running validate directly.")
        outputs = run_validation(paths, val_cfg, params_path)
        print("\n[HP30] Stage 3 validation outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
