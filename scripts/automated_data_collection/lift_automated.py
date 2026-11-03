"""Direct automated lift with osc_pose_abs_quat.

The arm receives one absolute TCP target at a time:
1. Send the TCP to the observed cube center.
2. Close the gripper once the observed TCP is close enough.
3. Lift through intermediate TCP waypoints toward the observed lift goal.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.automated_data_collection.utils.keyboard_gripper import (
        KeyboardGripperControl,
        add_keyboard_gripper_args,
        wrap_step_target_for_keyboard,
    )
    from scripts.automated_data_collection.utils.runner import GripperProfile
    from scripts.automated_data_collection.utils.target_noise import (
        TargetNoiseState,
        add_target_noise_args,
    )
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from utils.keyboard_gripper import (  # type: ignore[no-redef]
        KeyboardGripperControl,
        add_keyboard_gripper_args,
        wrap_step_target_for_keyboard,
    )
    from utils.runner import GripperProfile  # type: ignore[no-redef]
    from utils.target_noise import (  # type: ignore[no-redef]
        TargetNoiseState,
        add_target_noise_args,
    )
from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch

UMI_OPEN_ACTION = (1.0, -1.0)
UMI_CLOSE_ACTION = (-1.0, 1.0)


@dataclass(frozen=True)
class ActionSlices:
    """Flat action-vector slices for the arm and gripper segments."""

    total_dim: int
    arm: slice
    gripper: slice


@dataclass
class GraspTargetState:
    """Track the controlled descent into a grasp target."""

    descent_progress: float = 0.0


@dataclass(frozen=True)
class TcpToWristFrame:
    """Reset-time transform because osc_pose_abs_quat controls the wrist link."""

    wrist_origin_pos: torch.Tensor
    wrist_origin_quat: torch.Tensor
    tcp_from_wrist_pos: torch.Tensor

    @classmethod
    def from_obs(cls, obs: dict, device: torch.device) -> TcpToWristFrame:
        """Build the reset-time TCP-to-wrist offset from observations."""
        wrist_pos = _eef_pos(obs, device)
        wrist_quat = _eef_quat(obs, device)
        tcp_pos = _tcp_pos(obs, device)
        wrist_rot = quat_to_rotmat_wxyz_torch(wrist_quat)
        tcp_from_wrist_pos = (
            wrist_rot.transpose(-1, -2)
            .matmul(
                (tcp_pos - wrist_pos).unsqueeze(-1),
            )
            .squeeze(-1)
        )
        return cls(
            wrist_origin_pos=wrist_pos.detach().clone(),
            wrist_origin_quat=wrist_quat.detach().clone(),
            tcp_from_wrist_pos=tcp_from_wrist_pos.detach().clone(),
        )

    def wrist_pos_for_tcp(self, tcp_target_pos: torch.Tensor) -> torch.Tensor:
        """Return the wrist target position that places the TCP at `tcp_target_pos`."""
        wrist_rot = quat_to_rotmat_wxyz_torch(self.wrist_origin_quat)
        return tcp_target_pos - wrist_rot.matmul(
            self.tcp_from_wrist_pos.unsqueeze(-1),
        ).squeeze(-1)


class TraceRecorder:
    """Capture commanded targets, actions, observations, and Mink solve outputs."""

    def __init__(self, args: argparse.Namespace, env, slices: ActionSlices) -> None:
        """Create a trace directory and initialize per-step buffers."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.root = Path(args.trace_dir).expanduser() / f"{args.trace_prefix}_{stamp}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.env = env
        self.slices = slices
        self.rows: list[dict[str, Any]] = []
        self.last_solver: dict[str, Any] = {}
        self.meta: dict[str, Any] = {
            "script": "lift_automated.py",
            "task": args.task,
            "control_hz": int(args.control_hz),
            "grasp_z_offset": float(args.grasp_z_offset),
            "pregrasp_z_offset": float(args.pregrasp_z_offset),
            "descend_start_xy": float(args.descend_start_xy),
            "descend_end_xy": float(args.descend_end_xy),
            "descend_progress_step": float(args.descend_progress_step),
            "reach_tolerance": float(args.reach_tolerance),
            "move_to_cube_steps": int(args.move_to_cube_steps),
            "close_steps": int(args.close_steps),
            "settle_steps": int(args.settle_steps),
            "lift_steps": int(args.lift_steps),
            "lift_start_progress": float(args.lift_start_progress),
            "hold_steps": int(args.hold_steps),
            "contact_force_threshold": float(args.contact_force_threshold),
            "contact_min_alpha": float(args.contact_min_alpha),
            "target_noise_xy_std": float(args.target_noise_xy_std),
            "target_noise_z_std": float(args.target_noise_z_std),
            "target_noise_hold_steps": int(args.target_noise_hold_steps),
            "target_noise_max_xy": float(args.target_noise_max_xy),
            "target_noise_max_z": float(args.target_noise_max_z),
            "target_noise_phases": args.target_noise_phases,
            "target_noise_seed": args.target_noise_seed,
            "gripper_mode": args.gripper_mode,
            "keyboard_gripper_speed": float(args.keyboard_gripper_speed),
            "keyboard_gripper_initial_alpha": float(
                args.keyboard_gripper_initial_alpha
            ),
            "keyboard_gripper_invert": bool(args.keyboard_gripper_invert),
        }
        self._add_controller_meta(env)

    def _add_controller_meta(self, env) -> None:
        ctrl = getattr(getattr(env, "robot", None), "arm_ctrl", None)
        backend = getattr(ctrl, "_backend", None)
        self.meta["controller"] = {
            "name": type(ctrl).__name__ if ctrl is not None else None,
            "backend": getattr(ctrl, "backend", None),
            "normalized": getattr(ctrl, "normalized", None),
            "clip": getattr(ctrl, "clip", None),
            "lin_range": _json_value(getattr(ctrl, "lin_range", None)),
            "mink_dt": getattr(ctrl, "mink_dt", None),
            "mink_substeps": getattr(ctrl, "mink_substeps", None),
            "integration_dt": getattr(ctrl, "_integration_dt", None),
        }
        if backend is not None:
            self.meta["mink_backend"] = {
                "solver": getattr(backend, "solver", None),
                "frame_type": getattr(backend, "frame_type", None),
                "position_cost": _json_value(getattr(backend, "position_cost", None)),
                "orientation_cost": _json_value(
                    getattr(backend, "orientation_cost", None)
                ),
                "posture_cost": _json_value(getattr(backend, "posture_cost", None)),
                "damping_cost": _json_value(getattr(backend, "damping_cost", None)),
                "lm_damping": getattr(backend, "lm_damping", None),
                "qp_damping": getattr(backend, "qp_damping", None),
                "max_joint_step": getattr(backend, "max_joint_step", None),
                "configuration_limit_gain": getattr(
                    backend, "configuration_limit_gain", None
                ),
                "pos_tol": getattr(backend, "pos_tol", None),
                "rot_tol": getattr(backend, "rot_tol", None),
            }

    def install_solver_trace(self, env) -> None:
        """Wrap the Mink backend solver so each command records solve outputs."""
        ctrl = getattr(getattr(env, "robot", None), "arm_ctrl", None)
        backend = getattr(ctrl, "_backend", None)
        if backend is None or not hasattr(backend, "solve_targets"):
            self.meta["solver_trace"] = "unavailable"
            return

        original = backend.solve_targets
        recorder = self

        def wrapped_solve_targets(**kwargs):
            q_cmd = original(**kwargs)
            recorder.last_solver = {
                "iterations": int(kwargs.get("iterations", 0)),
                "integration_dt": float(kwargs.get("integration_dt", 0.0)),
                "q_init": _tensor_values(kwargs.get("q_init")),
                "pos_tgt": _tensor_values(kwargs.get("pos_tgt")),
                "quat_tgt": _tensor_values(kwargs.get("quat_tgt")),
                "q_cmd": _tensor_values(q_cmd),
            }
            return q_cmd

        backend.solve_targets = wrapped_solve_targets
        self.meta["solver_trace"] = "wrapped"

    def record(
        self,
        *,
        episode: int,
        step_idx: int,
        phase: str,
        obs_before: dict,
        obs_after: dict,
        target: torch.Tensor,
        action: torch.Tensor,
        grip_alpha: float,
        reward,
        terminated,
        truncated,
        info: dict,
    ) -> None:
        """Append one sampled environment step to the in-memory trace."""
        if step_idx % max(1, int(self.args.trace_every)) != 0:
            return

        device = target.device
        tcp_before = _tcp_pos(obs_before, device)
        tcp_after = _tcp_pos(obs_after, device)
        control_before = _control_pos(self.env, obs_before, device)
        control_after = _control_pos(self.env, obs_after, device)
        cube_before = _object_pos(obs_before, "cube_pos", device)
        cube_after = _object_pos(obs_after, "cube_pos", device)
        goal_after = _goal_pos(obs_after, device)
        eef_before = _eef_pos(obs_before, device)
        eef_after = _eef_pos(obs_after, device)
        arm_action = action[self.slices.arm]
        grip_action = action[self.slices.gripper]

        row: dict[str, Any] = {
            "episode": int(episode),
            "step": int(step_idx),
            "phase": phase,
            "grip_alpha": float(grip_alpha),
            "reward": _scalar(reward),
            "terminated": _bool_scalar(terminated),
            "truncated": _bool_scalar(truncated),
            "success": _bool_scalar(info.get("success", False)),
            "target_err_after": float(torch.linalg.norm(control_after - target)),
            "target_xy_err_after": float(
                torch.linalg.norm(control_after[:2] - target[:2])
            ),
            "target_z_err_after": float(control_after[2] - target[2]),
            "cube_xy_err_after": float(
                torch.linalg.norm(control_after[:2] - cube_after[:2])
            ),
            "cube_z_err_after": float(control_after[2] - cube_after[2]),
            "contact_after": float(_tactile_contact(obs_after, device)),
            "contact_force_after": float(_tactile_force_norm(obs_after, device)),
            "arm_action_norm": float(torch.linalg.norm(arm_action)),
            "arm_pos_action_norm": float(torch.linalg.norm(arm_action[:3])),
            "arm_quat_action_norm": float(torch.linalg.norm(arm_action[3:7])),
            "qpos_step_norm": float(
                torch.linalg.norm(
                    _state_vec(obs_after, "manipulator", "qpos", device)
                    - _state_vec(obs_before, "manipulator", "qpos", device),
                ),
            ),
        }
        _add_vec(row, "target", target)
        _add_vec(row, "tcp_before", tcp_before)
        _add_vec(row, "tcp_after", tcp_after)
        _add_vec(row, "control_before", control_before)
        _add_vec(row, "control_after", control_after)
        _add_vec(row, "cube_before", cube_before)
        _add_vec(row, "cube_after", cube_after)
        _add_vec(row, "goal_after", goal_after)
        _add_vec(row, "eef_before", eef_before)
        _add_vec(row, "eef_after", eef_after)
        _add_vec(row, "arm_action", arm_action)
        _add_vec(row, "grip_action", grip_action)
        _add_vec(
            row,
            "arm_qpos_before",
            _state_vec(obs_before, "manipulator", "qpos", device),
        )
        _add_vec(
            row, "arm_qpos_after", _state_vec(obs_after, "manipulator", "qpos", device)
        )
        _add_vec(
            row, "grip_qpos_after", _state_vec(obs_after, "gripper", "qpos", device)
        )

        solver = self.last_solver
        row["solver_iterations"] = solver.get("iterations", "")
        row["solver_integration_dt"] = solver.get("integration_dt", "")
        _add_list(row, "solver_pos_tgt", solver.get("pos_tgt", []))
        _add_list(row, "solver_quat_tgt", solver.get("quat_tgt", []))
        _add_list(row, "solver_q_init", solver.get("q_init", []))
        _add_list(row, "solver_q_cmd", solver.get("q_cmd", []))
        if solver.get("q_init") and solver.get("q_cmd"):
            q_init = torch.tensor(solver["q_init"], dtype=torch.float32)
            q_cmd = torch.tensor(solver["q_cmd"], dtype=torch.float32)
            n = min(q_init.numel(), q_cmd.numel())
            q_step = q_cmd[:n] - q_init[:n]
            row["solver_q_step_norm"] = float(torch.linalg.norm(q_step))
            row["solver_q_step_absmax"] = float(q_step.abs().max())

        self.rows.append(row)

    def save(self) -> list[Path]:
        """Write the trace CSV, metadata JSON, and optional summary plot."""
        csv_path = self.root / "trajectory.csv"
        meta_path = self.root / "metadata.json"
        plot_path = self.root / "trajectory_summary.png"

        keys = _csv_keys(self.rows)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        with meta_path.open("w") as f:
            json.dump(self.meta, f, indent=2, sort_keys=True)

        paths = [csv_path, meta_path]
        if self._save_plot(plot_path):
            paths.append(plot_path)
        return paths

    def _save_plot(self, path: Path) -> bool:
        if not self.rows:
            return False
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except (ImportError, OSError, RuntimeError):
            return False

        steps = [row["step"] for row in self.rows]
        phases = [row["phase"] for row in self.rows]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(
            steps, [row["target_err_after"] for row in self.rows], label="target err"
        )
        axes[0].plot(
            steps,
            [row["target_xy_err_after"] for row in self.rows],
            label="target xy err",
        )
        axes[0].legend(loc="best")
        axes[0].set_ylabel("meters")

        axes[1].plot(steps, [row["target_2"] for row in self.rows], label="target z")
        axes[1].plot(steps, [row["tcp_after_2"] for row in self.rows], label="tcp z")
        axes[1].plot(steps, [row["cube_after_2"] for row in self.rows], label="cube z")
        axes[1].legend(loc="best")
        axes[1].set_ylabel("z")

        axes[2].plot(
            steps,
            [row["arm_pos_action_norm"] for row in self.rows],
            label="arm pos action norm",
        )
        axes[2].plot(
            steps,
            [row.get("solver_q_step_absmax", 0.0) for row in self.rows],
            label="solver q step max",
        )
        axes[2].legend(loc="best")
        axes[2].set_ylabel("action/q")

        axes[3].plot(
            steps, [row["grip_alpha"] for row in self.rows], label="grip alpha"
        )
        axes[3].plot(
            steps, [row["cube_xy_err_after"] for row in self.rows], label="tcp-cube xy"
        )
        axes[3].plot(
            steps, [row["contact_after"] for row in self.rows], label="contact"
        )
        axes[3].legend(loc="best")
        axes[3].set_ylabel("grip / meters")
        axes[3].set_xlabel("step")

        last_phase = None
        for step, phase in zip(steps, phases, strict=False):
            if phase != last_phase:
                for ax in axes:
                    ax.axvline(step, color="0.8", linewidth=0.8)
                axes[0].text(
                    step,
                    axes[0].get_ylim()[1],
                    phase,
                    rotation=90,
                    va="top",
                    fontsize=8,
                )
                last_phase = phase

        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the lift collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift")
    parser.add_argument("--manipulator", default="franka")
    parser.add_argument("--gripper", default="umi")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--render-mode", choices=("human", "rgb_array", "none"), default="none"
    )
    parser.add_argument("--cameras", default="front,wrist,right_side")
    parser.add_argument("--modalities", default="rgb")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0165)
    parser.add_argument("--pregrasp-z-offset", type=float, default=0.100)
    parser.add_argument("--descend-start-xy", type=float, default=0.015)
    parser.add_argument("--descend-end-xy", type=float, default=0.006)
    parser.add_argument("--descend-progress-step", type=float, default=0.025)
    parser.add_argument("--reach-tolerance", type=float, default=0.025)
    parser.add_argument("--move-to-cube-steps", type=int, default=100)
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--lift-steps", type=int, default=50)
    parser.add_argument("--lift-start-progress", type=float, default=0.0)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--contact-force-threshold", type=float, default=1e-5)
    parser.add_argument("--contact-min-alpha", type=float, default=1.0)
    add_target_noise_args(parser)
    add_keyboard_gripper_args(parser)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--trace-prefix", default="lift_trace")
    parser.add_argument("--trace-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915
    """Run direct automated lift collection."""
    args = parse_args()
    if args.viewer:
        args.render_mode = "human"
    os.environ.setdefault(
        "PYOPENGL_PLATFORM", "glx" if args.render_mode == "human" else "egl"
    )

    env = _make_env(args)
    logger = None
    trace = None
    gripper_control = KeyboardGripperControl.from_args(args)
    original_step_target = _step_target
    try:
        gripper_control.start()
        globals()["_step_target"] = wrap_step_target_for_keyboard(
            original_step_target,
            gripper_control,
        )
        obs, _ = env.reset()
        if env.horizon is not None and int(args.max_steps) > int(env.horizon):
            env.horizon = int(args.max_steps)

        if not args.no_record:
            from dexsuite.utils.lerobot_logger import LeRobotLogger

            logger = LeRobotLogger(task=args.task, env=env, root_dir=args.dataset_root)

        slices = _action_slices(env)
        low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=env.device
        )
        high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=env.device
        )
        if args.trace_dir is not None:
            trace = TraceRecorder(args, env, slices)
            trace.install_solver_trace(env)

        successes = 0
        dropped = 0
        attempts = 0
        saved = 0
        max_attempts = args.max_attempts or max(args.episodes, args.episodes * 5)

        while successes < args.episodes and attempts < max_attempts:
            if attempts > 0:
                obs, _ = env.reset()
            episode = attempts
            attempts += 1

            frame = TcpToWristFrame.from_obs(obs, env.device)
            print(
                f"episode={episode} tcp={_fmt(_tcp_pos(obs, env.device))} "
                f"cube={_fmt(_cube_target(obs, args, env.device))} "
                f"goal={_fmt(_goal_pos(obs, env.device))}",
            )

            success, obs, saved_now = _run_episode(
                env,
                obs,
                args,
                frame,
                slices,
                low,
                high,
                logger,
                episode,
                trace,
            )
            saved += saved_now
            if success:
                successes += 1
                print(f"episode={episode} success")
            else:
                dropped += 1
                print(f"episode={episode} dropped")

        print(
            f"done successes={successes} dropped={dropped} attempts={attempts}"
            + (f" saved={saved}" if logger is not None else " no_record=True"),
        )
        if trace is not None:
            paths = trace.save()
            print("trace saved " + " ".join(str(path) for path in paths))
        return 0
    finally:
        if logger is not None:
            logger.close()
        globals()["_step_target"] = original_step_target
        gripper_control.close()
        env.close()


def _run_episode(  # noqa: C901, PLR0911, PLR0912
    env,
    obs: dict,
    args: argparse.Namespace,
    frame: TcpToWristFrame,
    slices: ActionSlices,
    low: torch.Tensor,
    high: torch.Tensor,
    logger,
    episode: int,
    trace: TraceRecorder | None,
) -> tuple[bool, dict, int]:
    step_idx = 0
    cube_target = _cube_target(obs, args, env.device)
    goal_target = _goal_pos(obs, env.device)
    grasp_state = GraspTargetState()
    target_noise = TargetNoiseState.from_args(args, episode=episode)

    reached = False
    for _ in range(int(args.move_to_cube_steps)):
        target = _dynamic_grasp_target(
            obs,
            args,
            cube_target,
            env.device,
            grasp_state,
            control_pos=_control_pos(env, obs, env.device),
        )
        target = target_noise.apply(target, "to_cube")
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            target,
            grip_alpha=0.0,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase="to_cube",
        )
        _log_step(env, args, episode, step_idx, "to_cube", obs, target, success)
        step_idx += 1
        if success:
            return (
                _finish_episode(logger, args, success=True),
                obs,
                int(logger is not None),
            )
        if _target_error(env, obs, cube_target, env.device) <= float(
            args.reach_tolerance
        ):
            reached = True
            break
        if step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success=False), obs, 0

    if not reached:
        return _finish_episode(logger, args, success=False), obs, 0

    for i in range(int(args.close_steps)):
        alpha = _smoothstep(i / max(float(args.close_steps - 1), 1.0))
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            cube_target,
            grip_alpha=alpha,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase="close",
        )
        _log_step(env, args, episode, step_idx, "close", obs, cube_target, success)
        step_idx += 1
        if success:
            return (
                _finish_episode(logger, args, success=True),
                obs,
                int(logger is not None),
            )
        if step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success=False), obs, 0

    for _ in range(int(args.settle_steps)):
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            cube_target,
            grip_alpha=1.0,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase="settle",
        )
        _log_step(env, args, episode, step_idx, "settle", obs, cube_target, success)
        step_idx += 1
        if success:
            return (
                _finish_episode(logger, args, success=True),
                obs,
                int(logger is not None),
            )
        if step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success=False), obs, 0

    lift_steps = max(int(args.lift_steps), 2)
    for i in range(lift_steps):
        target = _smooth_lift_target(
            cube_target,
            goal_target,
            i,
            lift_steps,
            start_progress=float(args.lift_start_progress),
        )
        target = target_noise.apply(target, "lift", final=i >= lift_steps - 1)
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            target,
            grip_alpha=1.0,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase="lift",
        )
        _log_step(env, args, episode, step_idx, "lift", obs, target, success)
        step_idx += 1
        if success:
            return (
                _finish_episode(logger, args, success=True),
                obs,
                int(logger is not None),
            )
        if step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success=False), obs, 0

    for _ in range(int(args.hold_steps)):
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            goal_target,
            grip_alpha=1.0,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase="hold",
        )
        _log_step(env, args, episode, step_idx, "hold", obs, goal_target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            break

    return (
        _finish_episode(logger, args, success=success),
        obs,
        int(success and logger is not None),
    )


def _step_target(
    env,
    obs: dict,
    slices: ActionSlices,
    low: torch.Tensor,
    high: torch.Tensor,
    logger,
    frame: TcpToWristFrame,
    tcp_target: torch.Tensor,
    *,
    grip_alpha: float,
    trace: TraceRecorder | None,
    episode: int,
    step_idx: int,
    phase: str,
    gripper_control: KeyboardGripperControl | None = None,
) -> tuple[dict, bool]:
    obs_before = obs
    effective_grip_alpha = (
        grip_alpha
        if gripper_control is None
        else gripper_control.alpha_for_step(grip_alpha)
    )
    profile = _cached_gripper_profile(env)
    tcp_target_for_wrist = profile.tcp_target_for_control_target(obs, tcp_target)
    action = torch.zeros(slices.total_dim, dtype=torch.float32, device=env.device)
    action[slices.arm] = _arm_action_for_tcp(frame, tcp_target_for_wrist)
    action[slices.gripper] = profile.action(effective_grip_alpha)
    action = torch.minimum(torch.maximum(action, low), high)
    next_obs, reward, terminated, truncated, info = env.step(action)
    if logger is not None:
        logger.add_step_pre(
            obs,
            action,
            reward,
            terminated,
            truncated,
            info,
        )
    success = bool(torch.as_tensor(info.get("success", False)).reshape(-1)[0])
    if trace is not None:
        trace.record(
            episode=episode,
            step_idx=step_idx,
            phase=phase,
            obs_before=obs_before,
            obs_after=next_obs,
            target=tcp_target,
            action=action,
            grip_alpha=effective_grip_alpha,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    return next_obs, success


def _arm_action_for_tcp(
    frame: TcpToWristFrame,
    tcp_target: torch.Tensor,
) -> torch.Tensor:
    wrist_pos = frame.wrist_pos_for_tcp(tcp_target)
    dq = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=tcp_target.device
    )
    return torch.cat((wrist_pos - frame.wrist_origin_pos, _quat_normalize(dq)), dim=0)


def _make_env(args: argparse.Namespace):
    import dexsuite as ds

    cameras = (
        None
        if args.cameras.strip().lower() in {"none", "off", "false"}
        else [part.strip() for part in args.cameras.split(",") if part.strip()]
    )
    modalities_arg = getattr(args, "modalities", "rgb")
    modalities = tuple(
        part.strip() for part in modalities_arg.split(",") if part.strip()
    )
    render_mode = None if args.render_mode == "none" else args.render_mode
    return ds.make(
        args.task,
        robot=ds.options.RobotOptions(
            type_of_robot="single",
            visualize_tcp=False,
            single=ds.options.ArmOptions(
                manipulator=args.manipulator,
                gripper=args.gripper,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                    config={
                        "normalized": False,
                        "clip": False,
                        "lin_range": [1.5, 1.0, 1.0],
                        "mink_substeps": 12,
                        "position_cost": 8.0,
                        "orientation_cost": 6.0,
                        "posture_cost": 0.02,
                        "damping_cost": 0.004,
                        "max_joint_step": 0.05,
                        "pos_tol": 0.001,
                        "rot_tol": 0.001,
                    },
                ),
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": True, "clip": False},
                ),
            ),
        ),
        sim=ds.options.SimOptions(control_hz=int(args.control_hz), n_envs=1),
        render_mode=render_mode,
        cameras=cameras,
        modalities=modalities,
    )


def _finish_episode(logger, args: argparse.Namespace, success: bool) -> bool:
    if logger is None:
        return success
    if success or args.save_all:
        logger.save_episode()
    else:
        logger.clear_buffer()
    return success


def _log_step(
    env,
    args: argparse.Namespace,
    episode: int,
    step_idx: int,
    phase: str,
    obs: dict,
    target: torch.Tensor,
    success: bool,
) -> None:
    if step_idx % max(1, int(args.print_every)) != 0:
        return
    tcp = _tcp_pos(obs, target.device)
    control = _control_pos(env, obs, target.device)
    cube = _cube_target(obs, args, target.device)
    print(
        f"episode={episode} step={step_idx} phase={phase} "
        f"control_err={float(torch.linalg.norm(control - target)):.4f} "
        f"tcp_err={float(torch.linalg.norm(tcp - target)):.4f} "
        f"control_cube={_fmt(control - cube)} tcp_cube={_fmt(tcp - cube)} "
        f"cube={_fmt(cube)} success={success}",
    )


def _target_error(
    env_or_obs,
    obs_or_target: dict | torch.Tensor,
    target_or_device: torch.Tensor | torch.device,
    device: torch.device | None = None,
) -> float:
    if device is None:
        obs = env_or_obs
        target = obs_or_target
        target_device = target_or_device
        return float(torch.linalg.norm(_tcp_pos(obs, target_device) - target))
    return float(
        torch.linalg.norm(
            _control_pos(env_or_obs, obs_or_target, device) - target_or_device
        )
    )


def _control_pos(env, obs: dict, device: torch.device) -> torch.Tensor:
    profile = _cached_gripper_profile(env)
    return profile.control_pos(obs).to(dtype=torch.float32, device=device)


def _cached_gripper_profile(env) -> GripperProfile:
    profile = getattr(env, "_dexsuite_collection_gripper_profile", None)
    if profile is not None:
        return profile
    gripper = getattr(getattr(env, "robot", None), "gripper", None)
    gripper_name = str(
        getattr(
            gripper,
            "_registry_name",
            getattr(gripper.__class__, "_registry_name", type(gripper).__name__),
        ),
    )
    profile = GripperProfile(env=env, gripper_name=gripper_name, task_name="")
    env._dexsuite_collection_gripper_profile = profile
    return profile


def _json_value(value) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _tensor_values(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _tensor_values(value) -> list[float]:
    if value is None:
        return []
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[0]
    return [float(v) for v in tensor.reshape(-1)]


def _scalar(value) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return float(tensor.reshape(-1)[0])


def _bool_scalar(value) -> bool:
    tensor = torch.as_tensor(value, dtype=torch.bool)
    return bool(tensor.reshape(-1)[0])


def _add_vec(row: dict[str, Any], prefix: str, value: torch.Tensor) -> None:
    for idx, item in enumerate(_tensor_values(value)):
        row[f"{prefix}_{idx}"] = item


def _add_list(row: dict[str, Any], prefix: str, value: list[float]) -> None:
    for idx, item in enumerate(value):
        row[f"{prefix}_{idx}"] = float(item)


def _csv_keys(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "episode",
        "step",
        "phase",
        "success",
        "reward",
        "target_err_after",
        "target_xy_err_after",
        "target_z_err_after",
        "cube_xy_err_after",
        "cube_z_err_after",
        "contact_after",
        "contact_force_after",
        "grip_alpha",
        "arm_action_norm",
        "arm_pos_action_norm",
        "solver_q_step_norm",
        "solver_q_step_absmax",
    ]
    keys = set()
    for row in rows:
        keys.update(row)
    return [key for key in preferred if key in keys] + sorted(
        keys.difference(preferred)
    )


def _dynamic_grasp_target(
    obs: dict,
    args: argparse.Namespace,
    grasp_target: torch.Tensor,
    device: torch.device,
    state: GraspTargetState,
    *,
    control_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    tcp = _tcp_pos(obs, device) if control_pos is None else control_pos
    xy_err = torch.linalg.norm(tcp[:2] - grasp_target[:2])
    start_xy = max(float(args.descend_start_xy), float(args.descend_end_xy) + 1e-6)
    end_xy = max(float(args.descend_end_xy), 0.0)
    u = torch.clamp((xy_err - end_xy) / (start_xy - end_xy), 0.0, 1.0)
    desired_progress = float(1.0 - _smootherstep_tensor(u))
    max_step = max(float(args.descend_progress_step), 1e-6)
    if desired_progress > state.descent_progress:
        state.descent_progress += min(
            desired_progress - state.descent_progress, max_step
        )
    state.descent_progress = float(max(0.0, min(1.0, state.descent_progress)))

    target = grasp_target.detach().clone()
    target[2] += float(args.pregrasp_z_offset) * (1.0 - state.descent_progress)
    return target


def _smooth_lift_target(
    start: torch.Tensor,
    goal: torch.Tensor,
    step: int,
    steps: int,
    *,
    start_progress: float,
) -> torch.Tensor:
    denom = max(int(steps), 1)
    t = torch.tensor(
        float(step + 1) / float(denom),
        dtype=start.dtype,
        device=start.device,
    )
    start_u = max(0.0, min(float(start_progress), 0.25))
    u = start_u + (1.0 - start_u) * _smootherstep_tensor(t)
    target = (1.0 - u) * start + u * goal
    if step >= denom - 1:
        target = goal.detach().clone()
    return target.contiguous()


def _cube_target(
    obs: dict, args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    target = _object_pos(obs, "cube_pos", device)
    target = target.detach().clone()
    target[2] += float(args.grasp_z_offset)
    return target


def _goal_pos(obs: dict, device: torch.device) -> torch.Tensor:
    return _object_pos(obs, "target_pos", device)


def _action_slices(env) -> ActionSlices:
    layout = getattr(getattr(env, "robot", None), "_layout", None)
    if layout is None:
        raise RuntimeError("Robot action layout is not initialized.")
    arm = None
    gripper = None
    for segment in layout.segments:
        name = ".".join(str(part) for part in segment.name).lower()
        if arm is None and ("manipulator" in name or "arm" in name):
            arm = slice(segment.start, segment.stop)
        if gripper is None and "gripper" in name:
            gripper = slice(segment.start, segment.stop)
    if arm is None or gripper is None:
        raise RuntimeError("Expected one arm action segment and one gripper segment.")
    return ActionSlices(int(layout.total_dim), arm, gripper)


def _has_tactile_contact(
    obs: dict,
    args: argparse.Namespace,
    device: torch.device,
    *,
    min_alpha: float,
) -> bool:
    if float(min_alpha) < float(args.contact_min_alpha):
        return False
    return _tactile_contact(obs, device) > 0.0 or (
        _tactile_force_norm(obs, device) >= float(args.contact_force_threshold)
    )


def _tactile_contact(obs: dict, device: torch.device) -> float:
    tactile = obs.get("state", {}).get("gripper", {}).get("tactile")
    if not isinstance(tactile, dict):
        return 0.0
    contact = 0.0
    for value in tactile.values():
        if isinstance(value, dict) and "contact" in value:
            contact = max(contact, float(_any_vec(value["contact"], device).max()))
    return contact


def _tactile_force_norm(obs: dict, device: torch.device) -> float:
    tactile = obs.get("state", {}).get("gripper", {}).get("tactile")
    if not isinstance(tactile, dict):
        return 0.0
    force_norm = 0.0
    for value in tactile.values():
        if isinstance(value, dict) and "force_norm" in value:
            force_norm = max(
                force_norm, float(_any_vec(value["force_norm"], device).max())
            )
    return force_norm


def _gripper_action(
    alpha: float, gripper_slice: slice, device: torch.device
) -> torch.Tensor:
    width = gripper_slice.stop - gripper_slice.start
    a = float(max(0.0, min(1.0, alpha)))
    if width == 2:
        open_action = torch.tensor(UMI_OPEN_ACTION, dtype=torch.float32, device=device)
        close_action = torch.tensor(
            UMI_CLOSE_ACTION, dtype=torch.float32, device=device
        )
        return torch.lerp(open_action, close_action, a)
    return torch.full((width,), 2.0 * a - 1.0, dtype=torch.float32, device=device)


def _tcp_pos(obs: dict, device: torch.device) -> torch.Tensor:
    return _vec(obs["state"]["gripper"]["tcp_pos"], 3, device)


def _eef_pos(obs: dict, device: torch.device) -> torch.Tensor:
    return _vec(obs["state"]["manipulator"]["eef_pos"], 3, device)


def _eef_quat(obs: dict, device: torch.device) -> torch.Tensor:
    return _quat_normalize(_vec(obs["state"]["manipulator"]["eef_quat"], 4, device))


def _object_pos(obs: dict, key: str, device: torch.device) -> torch.Tensor:
    return _vec(obs["state"]["other"][key], 3, device)


def _state_vec(obs: dict, group: str, key: str, device: torch.device) -> torch.Tensor:
    return _any_vec(obs["state"][group][key], device)


def _any_vec(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[0]
    return tensor.reshape(-1).contiguous()


def _vec(value: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[0]
    return tensor.reshape(-1)[:width].contiguous()


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)


def _smoothstep(x: float) -> float:
    x = float(max(0.0, min(1.0, x)))
    return x * x * (3.0 - 2.0 * x)


def _smootherstep_tensor(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _fmt(vec: torch.Tensor) -> str:
    values = [float(v) for v in vec.detach().cpu().reshape(-1)]
    return "[" + ",".join(f"{v:+.3f}" for v in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
