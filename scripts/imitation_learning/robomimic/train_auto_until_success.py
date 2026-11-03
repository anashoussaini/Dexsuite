"""Launch DexSuite robomimic auto trainings until rollout success is high enough.

This is intentionally a thin orchestration script around ``train.py``. It keeps
the training command visible, watches the robomimic log file for rollout
``Success_Rate``, and stops a run once it reaches the requested threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PYTHON = "/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python"
DEFAULT_OUTPUT_DIR = "../bc_trained_models"
SUCCESS_PATTERN = re.compile(r'"Success_Rate"\s*:\s*([-+0-9.eE]+)')


@dataclass(frozen=True)
class AutoDataset:
    """Dataset metadata needed to invoke the training wrapper."""

    dataset_name: str
    repo_id: str
    task: str
    h5_basename: str


@dataclass
class _MonitorState:
    latest_success: float | None = None
    best_success: float | None = None
    threshold_hit_at: float | None = None


AUTO_DATASETS: dict[str, AutoDataset] = {
    "reach_auto": AutoDataset(
        dataset_name="reach_auto",
        repo_id="collected_dataset/reach/reach_260622_034435",
        task="reach",
        h5_basename="reach_auto",
    ),
    "stack_auto": AutoDataset(
        dataset_name="stack_auto",
        repo_id="collected_dataset/stack/stack_260622_035837",
        task="stack",
        h5_basename="stack_auto",
    ),
    "pick_and_place_auto": AutoDataset(
        dataset_name="pick_and_place_auto",
        repo_id="collected_dataset/pick_place/pick_place_260622_053759",
        task="pick_place",
        h5_basename="pick_and_place_auto",
    ),
    "push_auto": AutoDataset(
        dataset_name="push_auto",
        repo_id="collected_dataset/push/push_260622_055500",
        task="push",
        h5_basename="push_auto",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _output_root(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _run_dirs(output_dir: Path, run_name: str) -> list[Path]:
    root = output_dir / run_name
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _latest_run_dir(output_dir: Path, run_name: str) -> Path | None:
    run_dirs = _run_dirs(output_dir, run_name)
    return run_dirs[-1] if run_dirs else None


def _success_rates(log_path: Path, start_offset: int = 0) -> list[float]:
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return []
    if start_offset > 0:
        text = text[start_offset:]
    return [float(match) for match in SUCCESS_PATTERN.findall(text)]


def _latest_success_rate(log_path: Path, start_offset: int = 0) -> float | None:
    rates = _success_rates(log_path, start_offset=start_offset)
    return rates[-1] if rates else None


def _best_success_rate(log_path: Path, start_offset: int = 0) -> float | None:
    rates = _success_rates(log_path, start_offset=start_offset)
    return max(rates) if rates else None


def _build_train_command(dataset: AutoDataset, args: argparse.Namespace) -> list[str]:
    run_name = args.name_template.format(dataset=dataset.dataset_name)
    cmd = [
        args.python,
        "-m",
        "scripts.imitation_learning.robomimic.train",
        "--dataset-name",
        dataset.dataset_name,
        "--repo-id",
        dataset.repo_id,
        "--task",
        dataset.task,
        "--h5-basename",
        dataset.h5_basename,
        "--config",
        "bc",
        "--sensors",
        "state",
        "--architecture",
        "mlp",
        "--rollout",
        "standard",
        "--name",
        run_name,
        "--output-dir",
        args.output_dir,
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--epoch-every-n-steps",
        str(args.epoch_every_n_steps),
        "--validation-epoch-every-n-steps",
        str(args.validation_epoch_every_n_steps),
        "--hdf5-normalize-obs",
        "--train-dataset-keys",
        "actions",
        "rewards",
        "dones",
        "--lr",
        str(args.lr),
        "--bc-gaussian",
        "--save-models",
        "--render-video",
        "--video-skip",
        str(args.video_skip),
        "--rollout-enabled",
        "--rollout-n",
        str(args.rollout_n),
        "--rollout-horizon",
        str(args.rollout_horizon),
        "--rollout-rate",
        str(args.rollout_rate),
        "--rollout-warmstart",
        str(args.rollout_warmstart),
        "--rollout-terminate-on-success",
    ]
    if args.force_convert:
        cmd.append("--force-convert")
    return cmd


def _terminate_process_group(process: subprocess.Popen[Any], timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    else:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    else:
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")


def _new_run_dir(
    output_root: Path,
    run_name: str,
    previous_run_dirs: set[Path],
) -> Path | None:
    candidates = [
        path for path in _run_dirs(output_root, run_name) if path not in previous_run_dirs
    ]
    return candidates[-1] if candidates else None


def _update_monitor_state(
    dataset: AutoDataset,
    run_dir: Path,
    args: argparse.Namespace,
    state: _MonitorState,
) -> None:
    log_path = run_dir / "logs" / "log.txt"
    success = _latest_success_rate(log_path)
    current_best_success = _best_success_rate(log_path)
    if current_best_success is not None:
        state.best_success = current_best_success

    if success is not None and success != state.latest_success:
        state.latest_success = success
        print(
            "[auto-train] "
            f"{dataset.dataset_name} latest Success_Rate={success:.3f}, "
            f"best={state.best_success:.3f}",
            flush=True,
        )

    if (
        state.best_success is not None
        and state.best_success >= args.threshold
        and state.threshold_hit_at is None
    ):
        state.threshold_hit_at = time.monotonic()
        print(
            "[auto-train] threshold reached; waiting "
            f"{args.stop_delay_seconds:.1f}s before stopping",
            flush=True,
        )


def _should_stop_for_threshold(args: argparse.Namespace, state: _MonitorState) -> bool:
    return (
        state.threshold_hit_at is not None
        and time.monotonic() - state.threshold_hit_at >= args.stop_delay_seconds
    )


def _run_one(dataset: AutoDataset, args: argparse.Namespace) -> dict[str, Any]:
    output_root = _output_root(args.output_dir)
    run_name = args.name_template.format(dataset=dataset.dataset_name)
    cmd = _build_train_command(dataset, args)
    previous_run_dirs = set(_run_dirs(output_root, run_name))

    print("\n" + "=" * 100, flush=True)
    print(f"[auto-train] starting {dataset.dataset_name}", flush=True)
    print("[auto-train] command:", " ".join(cmd), flush=True)

    process = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=_repo_root(),
        start_new_session=True,
    )

    run_dir: Path | None = None
    monitor_state = _MonitorState()

    try:
        while process.poll() is None:
            if run_dir is None:
                run_dir = _new_run_dir(output_root, run_name, previous_run_dirs)
                if run_dir is not None:
                    print(f"[auto-train] run dir: {run_dir}", flush=True)

            if run_dir is not None:
                _update_monitor_state(dataset, run_dir, args, monitor_state)
                if _should_stop_for_threshold(args, monitor_state):
                    _terminate_process_group(process, timeout=args.stop_timeout_seconds)
                    break

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("[auto-train] interrupted; stopping active training process", flush=True)
        _terminate_process_group(process, timeout=args.stop_timeout_seconds)
        raise

    return_code = process.wait()
    if run_dir is None:
        run_dir = _latest_run_dir(output_root, run_name)
    best_success = (
        _best_success_rate(run_dir / "logs" / "log.txt") if run_dir is not None else None
    )
    reached = best_success is not None and best_success >= args.threshold
    result = {
        "dataset": dataset.dataset_name,
        "task": dataset.task,
        "run_name": run_name,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "best_success_rate": best_success,
        "threshold": args.threshold,
        "threshold_reached": reached,
        "return_code": return_code,
    }
    print(f"[auto-train] result: {json.dumps(result, indent=2)}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train auto low-dimensional BC-Gaussian policies and stop each run "
            "once rollout success reaches the threshold."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(AUTO_DATASETS),
        default=list(AUTO_DATASETS),
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--name-template",
        default="{dataset}_bc_lowdim_gaussian",
        help="Python format string with {dataset}.",
    )
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epoch-every-n-steps", type=int, default=100)
    parser.add_argument("--validation-epoch-every-n-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--rollout-n", type=int, default=5)
    parser.add_argument("--rollout-horizon", type=int, default=400)
    parser.add_argument("--rollout-rate", type=int, default=100)
    parser.add_argument("--rollout-warmstart", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument(
        "--stop-delay-seconds",
        type=float,
        default=45.0,
        help="Delay after threshold hit so robomimic can finish checkpoint writes.",
    )
    parser.add_argument("--stop-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Regenerate HDF5 before training each dataset.",
    )
    parser.add_argument(
        "--summary-json",
        default="runs/imitation_learning/robomimic/auto_train_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    """Run all requested training jobs."""
    args = parse_args()
    results: list[dict[str, Any]] = []
    for task_name in args.tasks:
        result = _run_one(AUTO_DATASETS[task_name], args)
        results.append(result)
        _write_summary(_repo_root() / args.summary_json, results)
        if result["return_code"] != 0 and not result["threshold_reached"]:
            print(
                f"[auto-train] {task_name} exited before reaching threshold; stopping batch.",
                file=sys.stderr,
                flush=True,
            )
            return int(result["return_code"] or 1)

    print("\n[auto-train] complete", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
