from __future__ import annotations

if __name__ != "__main__":
    import importlib.util
    import sysconfig
    from pathlib import Path

    stdlib_socket = Path(sysconfig.get_path("stdlib")) / "socket.py"
    spec = importlib.util.spec_from_file_location(__name__, stdlib_socket)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib socket module from {stdlib_socket}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
else:
    import argparse
    import os
    import subprocess
    import sys
    import threading
    import time
    from datetime import datetime
    from pathlib import Path

    BASELINE_SCRIPTS = [
        "CurVer.py",
        "ablation_de_only.py",
        "ablation_ga_ls.py",
        "ablation_ls_only.py",
        "ablation_no_freq_weights.py",
        "ablation_no_reim_scaling.py",
    ]


    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run all baseline1 programs sequentially.")
        parser.add_argument(
            "--log-file",
            default="socket_run_log.txt",
            help="Output text file used to store combined stdout/stderr logs.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Starting random seed for batch execution.",
        )
        parser.add_argument(
            "--num-seeds",
            type=int,
            default=10,
            help="Number of consecutive seeds to run.",
        )
        return parser


    def timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    def format_progress(current: int, total: int, width: int = 24) -> str:
        if total <= 0:
            total = 1
        ratio = min(max(current / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {current}/{total}"


    def print_status(message: str, current: int | None = None, total: int | None = None) -> None:
        prefix = f"{format_progress(current, total)} " if current is not None and total is not None else ""
        print(f"[{timestamp()}] {prefix}{message}", flush=True)


    def stream_reader(stream, sink: list[str], state: dict[str, float]) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line)
            state["last_output_time"] = time.monotonic()
        stream.close()


    def run_script(
        script_path: Path,
        python_executable: str,
        seed: int,
        current_step: int,
        total_steps: int,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        process = subprocess.Popen(
            [python_executable, str(script_path), "--no-show", "--seed", str(seed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_path.parent.parent),
            env=env,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError(f"Failed to capture output for {script_path.name}")

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        state = {"last_output_time": time.monotonic()}

        stdout_thread = threading.Thread(
            target=stream_reader,
            args=(process.stdout, stdout_lines, state),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=stream_reader,
            args=(process.stderr, stderr_lines, state),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:
            time.sleep(5)
            idle_seconds = time.monotonic() - state["last_output_time"]
            if idle_seconds >= 30:
                print_status(
                    f"running seed={seed} script={script_path.name} idle={int(idle_seconds)}s",
                    current=current_step,
                    total=total_steps,
                )
                state["last_output_time"] = time.monotonic()

        stdout_thread.join()
        stderr_thread.join()
        return subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )


    def main() -> int:
        args = build_arg_parser().parse_args()
        baseline_dir = Path(__file__).resolve().parent
        log_path = baseline_dir / args.log_file
        seeds = [args.seed + i for i in range(args.num_seeds)]
        total_steps = len(seeds) * len(BASELINE_SCRIPTS)

        lines: list[str] = []
        lines.append(f"socket.py run started at {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Python executable: {sys.executable}")
        lines.append(f"Starting seed: {args.seed}")
        lines.append(f"Number of seeds: {args.num_seeds}")
        lines.append(f"Seed list: {', '.join(str(seed) for seed in seeds)}")
        lines.append("")

        exit_code = 0
        print_status("starting baseline1 batch run")
        print_status(f"log file: {log_path}")
        print_status(f"seed list: {', '.join(str(seed) for seed in seeds)}")
        completed_steps = 0
        for run_index, seed in enumerate(seeds, start=1):
            print_status(f"starting seed {seed} ({run_index}/{len(seeds)})", current=completed_steps, total=total_steps)
            lines.append("#" * 80)
            lines.append(f"Seed batch {run_index}/{len(seeds)}")
            lines.append(f"Seed: {seed}")
            lines.append(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
            lines.append("#" * 80)
            for script_name in BASELINE_SCRIPTS:
                script_path = baseline_dir / script_name
                step_index = completed_steps + 1
                print_status(
                    f"starting seed={seed} script={script_name}",
                    current=step_index,
                    total=total_steps,
                )
                started_at = time.monotonic()
                result = run_script(script_path, sys.executable, seed, step_index, total_steps)
                duration = time.monotonic() - started_at
                status = "ok" if result.returncode == 0 else f"failed(code={result.returncode})"
                print_status(
                    f"finished seed={seed} script={script_name} {status} duration={duration:.1f}s",
                    current=step_index,
                    total=total_steps,
                )
                lines.append("=" * 80)
                lines.append(f"Script: {script_name}")
                lines.append(f"Seed: {seed}")
                lines.append(f"Return code: {result.returncode}")
                lines.append(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
                lines.append(f"Duration seconds: {duration:.3f}")
                lines.append("-" * 80)
                stdout = result.stdout.rstrip()
                stderr = result.stderr.rstrip()
                lines.append("[STDOUT]")
                lines.append(stdout if stdout else "<empty>")
                lines.append("")
                lines.append("[STDERR]")
                lines.append(stderr if stderr else "<empty>")
                lines.append("")
                if result.returncode != 0:
                    exit_code = result.returncode
                    stderr_preview = stderr.splitlines()[:3]
                    if stderr_preview:
                        print_status(
                            f"error summary for {script_name}: {' | '.join(stderr_preview)}",
                            current=step_index,
                            total=total_steps,
                        )
                completed_steps = step_index

        log_path.write_text("\n".join(lines), encoding="utf-8")
        print_status(f"combined log written to: {log_path}", current=completed_steps, total=total_steps)
        return exit_code


    if __name__ == "__main__":
        raise SystemExit(main())
