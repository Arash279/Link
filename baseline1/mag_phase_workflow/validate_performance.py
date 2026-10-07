from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline1 import CurVer
from baseline1.workflow.config import Paths
from baseline1.workflow.data import frame_to_arrays, load_experiment, mag_phase_to_complex
from baseline1.workflow.model import PARAM_NAMES, simulate_complex, wrap_phase_deg
from baseline1.workflow.validation import load_params


WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = WORKFLOW_DIR / "outputs"


def latest_params_file(output_dir: Path, table: str, seed: int) -> Path:
    exact = output_dir / f"mag_phase_fit_{table}_seed{seed}_params.json"
    if exact.exists():
        return exact

    matches = sorted(
        output_dir.glob(f"mag_phase_fit_{table}_seed*_params.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No mag/phase parameter file found in {output_dir} for table={table}.")


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
    parser = argparse.ArgumentParser(description="Validate log-magnitude + phase fit performance.")
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
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    params_path = Path(args.params_path) if args.params_path else latest_params_file(output_dir, args.table, args.seed)
    image_path = (
        Path(args.output_image)
        if args.output_image
        else output_dir / f"mag_phase_performance_{args.table}_seed{args.seed}.png"
    )

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
        title=f"Mag/Phase Fit Performance: {args.table}",
        output_path=image_path,
        show=args.show,
    )

    print("Mag/phase performance validation")
    print(f"table = {args.table}")
    print(f"params = {params_path}")
    print(f"image = {image_path}")
    print(f"SSE_raw = {metrics['SSE_raw']:.6g}")
    print(f"RMSE_raw = {metrics['RMSE_raw']:.6g}")
    print(f"AIC_raw = {metrics['AIC_raw']:.3f}")
    print(f"BIC_raw = {metrics['BIC_raw']:.3f}")
    print(f"n = {int(metrics['n'])}, p = {int(metrics['p'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

