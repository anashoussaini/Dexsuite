"""Run and record trajectory smoothness for the five single-arm baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

try:
    from scripts.automated_data_collection import (
        lift_automated as lift,
        pick_place_automated as pick_place,
        push_automated as push,
        stack_automated as stack,
    )
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    import lift_automated as lift  # type: ignore[no-redef]
    import pick_place_automated as pick_place  # type: ignore[no-redef]
    import push_automated as push  # type: ignore[no-redef]
    import stack_automated as stack  # type: ignore[no-redef]

DIRECT_TASKS: dict[str, ModuleType] = {
    "lift": lift,
    "pick_place": pick_place,
    "push": push,
    "stack": stack,
}


class StepRecorder:
    """Record executed target/action/observation deltas for one task."""

    def __init__(self, task: str, out_dir: Path) -> None:
        """Initialize one CSV recorder for a benchmarked task."""
        self.task = task
        self.rows: list[dict[str, Any]] = []
        self.out_dir = out_dir
        self._prev: dict[int, dict[str, torch.Tensor]] = {}

    def reset_episode(self, episode: int) -> None:
        """Clear previous-step state for one episode."""
        self._prev.pop(int(episode), None)

    def wrap_step_target(self, fn: Callable) -> Callable:
        """Wrap a direct task `_step_target` function to record sent actions."""
        def wrapped(
            env,
            obs: dict,
            slices,
            low: torch.Tensor,
            high: torch.Tensor,
            logger,
            frame,
            tcp_target: torch.Tensor,
            *,
            grip_alpha: float,
            trace,
            episode: int,
            step_idx: int,
            phase: str,
        ):
            action = _action_for_target(env, slices, low, high, frame, tcp_target, grip_alpha)
            next_obs, success = fn(
                env,
                obs,
                slices,
                low,
                high,
                logger,
                frame,
                tcp_target,
                grip_alpha=grip_alpha,
                trace=trace,
                episode=episode,
                step_idx=step_idx,
                phase=phase,
            )
            self.record(
                episode=episode,
                step_idx=step_idx,
                phase=phase,
                obs=next_obs,
                target=tcp_target,
                action=action,
                grip_alpha=grip_alpha,
                success=success,
            )
            return next_obs, success

        return wrapped

    def record(
        self,
        *,
        episode: int,
        step_idx: int,
        phase: str,
        obs: dict,
        target: torch.Tensor,
        action: torch.Tensor,
        grip_alpha: float,
        success: bool,
    ) -> None:
        """Append one executed step and local smoothness metrics."""
        episode = int(episode)
        tcp = lift._tcp_pos(obs, target.device)
        prev = self._prev.get(episode)
        action_delta = _delta_norm(action, None if prev is None else prev["action"])
        arm_delta = _delta_norm(action[:7], None if prev is None else prev["action"][:7])
        gripper_delta = _delta_norm(action[7:], None if prev is None else prev["action"][7:])
        target_delta = _delta_norm(target, None if prev is None else prev["target"])
        tcp_delta_vec = torch.zeros_like(tcp) if prev is None else tcp - prev["tcp"]
        tcp_step = float(torch.linalg.norm(tcp_delta_vec))
        tcp_accel_step = 0.0
        if prev is not None:
            tcp_accel_step = float(torch.linalg.norm(tcp_delta_vec - prev["tcp_delta"]))

        row: dict[str, Any] = {
            "task": self.task,
            "episode": episode,
            "step": int(step_idx),
            "phase": phase,
            "success": bool(success),
            "grip_alpha": float(grip_alpha),
            "target_delta": target_delta,
            "tcp_step": tcp_step,
            "tcp_accel_step": tcp_accel_step,
            "action_delta": action_delta,
            "arm_delta": arm_delta,
            "gripper_delta": gripper_delta,
            "target_err": float(torch.linalg.norm(tcp - target)),
        }
        _add_vec(row, "target", target)
        _add_vec(row, "tcp", tcp)
        _add_vec(row, "action", action)
        self._add_other_positions(row, obs, target.device)
        self.rows.append(row)
        self._prev[episode] = {
            "action": action.detach().clone(),
            "target": target.detach().clone(),
            "tcp": tcp.detach().clone(),
            "tcp_delta": tcp_delta_vec.detach().clone(),
        }

    def _add_other_positions(self, row: dict[str, Any], obs: dict, device: torch.device) -> None:
        other = obs.get("state", {}).get("other", {})
        if not isinstance(other, dict):
            return
        for key, value in sorted(other.items()):
            if not key.endswith("_pos"):
                continue
            vec = torch.as_tensor(value, dtype=torch.float32, device=device)
            if vec.ndim == 2:
                vec = vec[0]
            vec = vec.reshape(-1)
            if vec.numel() >= 3:
                _add_vec(row, key, vec[:3])

    def write_csv(self) -> Path:
        """Write recorded rows to the per-task trajectory CSV."""
        path = self.out_dir / f"{self.task}_trajectory.csv"
        keys = _csv_keys(self.rows)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def summary(self, successes: int, attempts: int) -> dict[str, Any]:
        """Return aggregate success and smoothness metrics."""
        return {
            "successes": int(successes),
            "attempts": int(attempts),
            "steps": len(self.rows),
            "max_target_delta": _max(self.rows, "target_delta"),
            "p95_target_delta": _percentile(self.rows, "target_delta", 0.95),
            "max_tcp_step": _max(self.rows, "tcp_step"),
            "p95_tcp_step": _percentile(self.rows, "tcp_step", 0.95),
            "max_tcp_accel_step": _max(self.rows, "tcp_accel_step"),
            "p95_tcp_accel_step": _percentile(self.rows, "tcp_accel_step", 0.95),
            "max_arm_delta": _max(self.rows, "arm_delta"),
            "p95_arm_delta": _percentile(self.rows, "arm_delta", 0.95),
            "max_gripper_delta": _max(self.rows, "gripper_delta"),
            "spike_count": _spike_count(self.rows),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline benchmarking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--tasks", default="reach,lift,pick_place,push,stack")
    parser.add_argument("--print-every", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """Run selected baseline collectors and write trajectory summaries."""
    args = parse_args()
    out_dir = args.out_dir or Path(tempfile.gettempdir()) / (
        f"dexsuite_baseline_trajectory_check_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    summaries: dict[str, Any] = {"out_dir": str(out_dir), "tasks": {}}
    for task in [part.strip() for part in args.tasks.split(",") if part.strip()]:
        print(f"task={task} starting episodes={args.episodes}")
        if task == "reach":
            summary = _run_reach(args, out_dir)
        elif task in DIRECT_TASKS:
            summary = _run_direct_task(task, DIRECT_TASKS[task], args, out_dir)
        else:
            raise ValueError(f"Unknown task {task!r}.")
        summaries["tasks"][task] = summary
        print(
            f"task={task} successes={summary['successes']}/{summary['attempts']} "
            f"max_tcp_step={summary['max_tcp_step']:.4f} "
            f"p95_tcp_step={summary['p95_tcp_step']:.4f} "
            f"spikes={summary['spike_count']}",
        )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(f"summary={summary_path}")
    return 0


def _run_direct_task(
    task: str,
    module: ModuleType,
    bench_args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    args = _module_default_args(module)
    args.episodes = 1
    args.max_attempts = 1
    args.no_record = True
    args.render_mode = "none"
    args.cameras = "none"
    args.control_hz = int(bench_args.control_hz)
    if int(bench_args.print_every) > 0:
        args.print_every = int(bench_args.print_every)
    else:
        args.print_every = 1_000_000

    env = lift._make_env(args)
    recorder = StepRecorder(task, out_dir)
    original_step_target = module._step_target
    module._step_target = recorder.wrap_step_target(original_step_target)
    try:
        obs, _ = env.reset()
        if env.horizon is not None and int(args.max_steps) > int(env.horizon):
            env.horizon = int(args.max_steps)
        slices = lift._action_slices(env)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)

        successes = 0
        attempts = 0
        for episode in range(int(bench_args.episodes)):
            if episode > 0:
                obs, _ = env.reset()
            recorder.reset_episode(episode)
            frame = lift.TcpToWristFrame.from_obs(obs, env.device)
            success, obs, _ = _call_run_episode(
                task,
                module,
                env,
                obs,
                args,
                frame,
                slices,
                low,
                high,
                episode,
            )
            attempts += 1
            successes += int(success)

        csv_path = recorder.write_csv()
        summary = recorder.summary(successes, attempts)
        summary["csv"] = str(csv_path)
        return summary
    finally:
        module._step_target = original_step_target
        env.close()


def _call_run_episode(
    task: str,
    module: ModuleType,
    env,
    obs: dict,
    args: argparse.Namespace,
    frame: lift.TcpToWristFrame,
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    episode: int,
) -> tuple[bool, dict, int]:
    if task == "lift":
        return module._run_episode(env, obs, args, frame, slices, low, high, None, episode, None)
    return module._run_episode(env, obs, args, frame, slices, low, high, None, episode)


def _run_reach(bench_args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    args = SimpleNamespace(
        task="reach",
        manipulator="franka",
        gripper="umi",
        max_steps=180,
        control_hz=int(bench_args.control_hz),
        render_mode="none",
        cameras="none",
        reach_steps=130,
        hold_steps=10,
        start_progress=0.02,
        reach_tolerance=0.004,
        no_record=True,
        save_all=False,
    )
    env = lift._make_env(args)
    recorder = StepRecorder("reach", out_dir)
    step_target = recorder.wrap_step_target(lift._step_target)
    try:
        obs, _ = env.reset()
        if env.horizon is not None and int(args.max_steps) > int(env.horizon):
            env.horizon = int(args.max_steps)
        slices = lift._action_slices(env)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)

        successes = 0
        attempts = 0
        for episode in range(int(bench_args.episodes)):
            if episode > 0:
                obs, _ = env.reset()
            recorder.reset_episode(episode)
            frame = lift.TcpToWristFrame.from_obs(obs, env.device)
            success, obs = _run_reach_episode(
                env,
                obs,
                args,
                frame,
                slices,
                low,
                high,
                step_target,
                episode,
            )
            attempts += 1
            successes += int(success)

        csv_path = recorder.write_csv()
        summary = recorder.summary(successes, attempts)
        summary["csv"] = str(csv_path)
        return summary
    finally:
        env.close()


def _run_reach_episode(
    env,
    obs: dict,
    args: SimpleNamespace,
    frame: lift.TcpToWristFrame,
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    step_target: Callable,
    episode: int,
) -> tuple[bool, dict]:
    start = lift._tcp_pos(obs, env.device).detach().clone()
    goal = lift._object_pos(obs, "target_pos", env.device).detach().clone()
    success = False
    step_idx = 0
    for i in range(int(args.reach_steps)):
        target = _smooth_line(start, goal, i, int(args.reach_steps), start_progress=float(args.start_progress))
        obs, success = step_target(
            env,
            obs,
            slices,
            low,
            high,
            None,
            frame,
            target,
            grip_alpha=0.0,
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="reach:target",
        )
        step_idx += 1
        if success or float(torch.linalg.norm(lift._tcp_pos(obs, env.device) - goal)) <= float(args.reach_tolerance):
            return True, obs

    for _ in range(int(args.hold_steps)):
        obs, success = step_target(
            env,
            obs,
            slices,
            low,
            high,
            None,
            frame,
            goal,
            grip_alpha=0.0,
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="reach:hold",
        )
        step_idx += 1
        if success:
            return True, obs
    return success, obs


def _action_for_target(
    env,
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    frame: lift.TcpToWristFrame,
    tcp_target: torch.Tensor,
    grip_alpha: float,
) -> torch.Tensor:
    action = torch.zeros(slices.total_dim, dtype=torch.float32, device=env.device)
    action[slices.arm] = lift._arm_action_for_tcp(frame, tcp_target)
    action[slices.gripper] = lift._gripper_action(grip_alpha, slices.gripper, env.device)
    return torch.minimum(torch.maximum(action, low), high).detach().clone()


def _module_default_args(module: ModuleType) -> argparse.Namespace:
    with _temporary_argv([str(Path(module.__file__ or module.__name__))]):
        return module.parse_args()


@contextmanager
def _temporary_argv(argv: list[str]):
    old = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old


def _smooth_line(
    start: torch.Tensor,
    goal: torch.Tensor,
    step: int,
    steps: int,
    *,
    start_progress: float,
) -> torch.Tensor:
    denom = max(int(steps), 1)
    t = float(step + 1) / float(denom)
    start_u = max(0.0, min(float(start_progress), 0.25))
    u = start_u + (1.0 - start_u) * lift._smoothstep(t)
    target = (1.0 - u) * start + u * goal
    if step >= denom - 1:
        target = goal.detach().clone()
    return target.contiguous()


def _delta_norm(value: torch.Tensor, prev: torch.Tensor | None) -> float:
    if prev is None:
        return 0.0
    n = min(value.numel(), prev.numel())
    return float(torch.linalg.norm(value.reshape(-1)[:n] - prev.reshape(-1)[:n]))


def _add_vec(row: dict[str, Any], prefix: str, value: torch.Tensor) -> None:
    for idx, item in enumerate(value.detach().cpu().reshape(-1)):
        row[f"{prefix}_{idx}"] = float(item)


def _csv_keys(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "task",
        "episode",
        "step",
        "phase",
        "success",
        "target_err",
        "target_delta",
        "tcp_step",
        "tcp_accel_step",
        "action_delta",
        "arm_delta",
        "gripper_delta",
        "grip_alpha",
    ]
    keys = set()
    for row in rows:
        keys.update(row)
    return [key for key in preferred if key in keys] + sorted(keys.difference(preferred))


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if key in row and not bool(row.get("success", False))
    ]


def _max(rows: list[dict[str, Any]], key: str) -> float:
    values = _values(rows, key)
    return max(values) if values else 0.0


def _percentile(rows: list[dict[str, Any]], key: str, q: float) -> float:
    values = sorted(_values(rows, key))
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round(float(q) * (len(values) - 1))))
    return values[idx]


def _spike_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if bool(row.get("success", False)):
            continue
        if (
            float(row.get("tcp_step", 0.0)) > 0.040
            or float(row.get("tcp_accel_step", 0.0)) > 0.030
            or float(row.get("arm_delta", 0.0)) > 0.080
        ):
            count += 1
    return count


if __name__ == "__main__":
    raise SystemExit(main())
