from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from baseline1 import CurVer
    from baseline1.hp30_workflow.config import Paths
    from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex
    from baseline1.workflow.model import PARAM_NAMES, simulate_complex, wrap_phase_deg
    from baseline1.workflow.validation import load_params
else:
    from baseline1 import CurVer
    from .config import Paths
    from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex
    from baseline1.workflow.model import PARAM_NAMES, simulate_complex, wrap_phase_deg
    from baseline1.workflow.validation import load_params


def latest_params_file(output_dir: Path, table: str, seed: int) -> Path:
    exact = output_dir / f"stage2_fit_params_{table}_seed{seed}.json"
    if exact.exists():
        return exact
    matches = sorted(
        output_dir.glob(f"stage2_fit_params_{table}_seed*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No fitted parameter file found in {output_dir} for table={table}.")


def plot_impedance_compare(
    f_hz: np.ndarray,
    z_exp: np.ndarray,
    z_sim: np.ndarray,
    title: str,
    output_path: Path,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mag_exp = np.abs(z_exp)
    mag_sim = np.abs(z_sim)
    phase_exp = wrap_phase_deg(np.angle(z_exp, deg=True))
    phase_sim = wrap_phase_deg(np.angle(z_sim, deg=True))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].semilogx(f_hz, mag_exp, label="Experiment", linewidth=1.8)
    axes[0].semilogx(f_hz, mag_sim, "--", label="Simulation", linewidth=1.8)
    axes[0].set_ylabel("|Z| (Ohm)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.55)
    axes[0].legend()

    axes[1].semilogx(f_hz, phase_exp, label="Experiment", linewidth=1.8)
    axes[1].semilogx(f_hz, phase_sim, "--", label="Simulation", linewidth=1.8)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.55)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate fitted HP_30 performance with a saved parameter file.")
    parser.add_argument("--table", default="exp_10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params-path", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--max-freq", type=float, default=1e8)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = Paths()
    db_path = Path(args.db_path) if args.db_path else paths.db_path
    output_dir = Path(args.output_dir) if args.output_dir else paths.output_dir
    params_path = Path(args.params_path) if args.params_path else latest_params_file(output_dir, args.table, args.seed)
    image_path = Path(args.output_image) if args.output_image else output_dir / f"stage3_performance_{args.table}_seed{args.seed}.png"

    params = load_params(params_path)
    exp = load_experiment(db_path, args.table, max_freq=args.max_freq)
    f_hz, zabs, phase = frame_to_arrays(exp)
    z_exp = mag_phase_to_complex(zabs, phase)
    z_sim = simulate_complex(f_hz, params)
    metrics = CurVer.evaluate_raw_space_metrics(z_sim, z_exp, len(PARAM_NAMES))

    plot_impedance_compare(
        f_hz,
        z_exp,
        z_sim,
        title=f"HP_30 Fit Performance: {args.table}",
        output_path=image_path,
        show=args.show,
    )

    print("[HP30] Performance validation")
    print(f"[HP30] db = {db_path}")
    print(f"[HP30] table = {args.table}")
    print(f"[HP30] params = {params_path}")
    print(f"[HP30] image = {image_path}")
    print(f"[HP30] SSE_raw = {metrics['SSE_raw']:.6g}")
    print(f"[HP30] RMSE_raw = {metrics['RMSE_raw']:.6g}")
    print(f"[HP30] AIC_raw = {metrics['AIC_raw']:.3f}")
    print(f"[HP30] BIC_raw = {metrics['BIC_raw']:.3f}")
    print(f"[HP30] n = {int(metrics['n'])}, p = {int(metrics['p'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
