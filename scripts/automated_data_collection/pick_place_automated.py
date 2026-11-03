"""Direct automated pick-place with osc_pose_abs_quat."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

try:
    from scripts.automated_data_collection.lift_automated import (
        GraspTargetState,
        TcpToWristFrame,
        _action_slices,
        _control_pos,
        _dynamic_grasp_target,
        _finish_episode,
        _fmt,
        _make_env,
        _object_pos,
        _smoothstep,
        _step_target,
        _target_error,
        _tcp_pos,
    )
    from scripts.automated_data_collection.utils.keyboard_gripper import (
        KeyboardGripperControl,
        add_keyboard_gripper_args,
        wrap_step_target_for_keyboard,
    )
    from scripts.automated_data_collection.utils.target_noise import (
        TargetNoiseState,
        add_target_noise_args,
    )
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from lift_automated import (  # type: ignore[no-redef]
        GraspTargetState,
        TcpToWristFrame,
        _action_slices,
        _control_pos,
        _dynamic_grasp_target,
        _finish_episode,
        _fmt,
        _make_env,
        _object_pos,
        _smoothstep,
        _step_target,
        _target_error,
        _tcp_pos,
    )
    from utils.keyboard_gripper import (  # type: ignore[no-redef]
        KeyboardGripperControl,
        add_keyboard_gripper_args,
        wrap_step_target_for_keyboard,
    )
    from utils.target_noise import (  # type: ignore[no-redef]
        TargetNoiseState,
        add_target_noise_args,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pick-place collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pick_place")
    parser.add_argument("--manipulator", default="franka")
    parser.add_argument("--gripper", default="umi")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render-mode", choices=("human", "rgb_array", "none"), default="none")
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
    parser.add_argument("--transport-steps", type=int, default=40)
    parser.add_argument("--transport-arc-height", type=float, default=0.095)
    parser.add_argument("--transport-start-progress", type=float, default=0.0)
    parser.add_argument("--release-cube-z-offset", type=float, default=0.016)
    parser.add_argument("--transport-release-start", type=float, default=None)
    parser.add_argument("--release-steps", type=int, default=5)
    parser.add_argument("--retreat-steps", type=int, default=8)
    parser.add_argument("--retreat-height", type=float, default=0.080)
    parser.add_argument("--contact-force-threshold", type=float, default=1e-5)
    parser.add_argument("--contact-min-alpha", type=float, default=1.0)
    add_target_noise_args(parser)
    add_keyboard_gripper_args(parser)
    return parser.parse_args()


def main() -> int:
    """Run direct automated pick-place collection."""
    args = parse_args()
    if args.viewer:
        args.render_mode = "human"
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx" if args.render_mode == "human" else "egl")

    env = _make_env(args)
    logger = None
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
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)

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
                f"goal={_fmt(_place_cube_center(obs, args, env.device))}",
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
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    logger,
    episode: int,
) -> tuple[bool, dict, int]:
    step_idx = 0
    cube_target = _cube_target(obs, args, env.device)
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
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="to_cube",
        )
        _log_step(args, episode, step_idx, "to_cube", obs, target, success)
        step_idx += 1
        if success:
            return _finish_episode(logger, args, success=True), obs, int(logger is not None)
        if _target_error(env, obs, cube_target, env.device) <= float(args.reach_tolerance):
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
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="close",
        )
        _log_step(args, episode, step_idx, "close", obs, cube_target, success)
        step_idx += 1
        if success:
            return _finish_episode(logger, args, success=True), obs, int(logger is not None)
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
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="settle",
        )
        _log_step(args, episode, step_idx, "settle", obs, cube_target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    control_from_cube = (
        _control_pos(env, obs, env.device) - _object_pos(obs, "cube_pos", env.device)
    ).detach().clone()
    place_center = _place_cube_center(obs, args, env.device)
    release_target = place_center + control_from_cube
    retreat_target = release_target.detach().clone()
    retreat_target[2] += float(args.retreat_height)

    obs, step_idx, success = _run_arc_segment(
        env,
        obs,
        args,
        frame,
        slices,
        low,
        high,
        logger,
        episode,
        step_idx,
        "transport",
        cube_target,
        release_target,
        int(args.transport_steps),
        grip_alpha=1.0,
        arc_height=float(args.transport_arc_height),
        start_progress=float(args.transport_start_progress),
        release_grip_start=args.transport_release_start,
        target_noise=target_noise,
    )
    if success or step_idx >= int(args.max_steps):
        return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    release_steps = max(int(args.release_steps), 1)
    for i in range(release_steps):
        alpha = 1.0 - _smoothstep(i / max(float(release_steps - 1), 1.0))
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            release_target,
            grip_alpha=alpha,
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="release",
        )
        _log_step(args, episode, step_idx, "release", obs, release_target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    obs, step_idx, success = _run_smooth_segment(
        env,
        obs,
        args,
        frame,
        slices,
        low,
        high,
        logger,
        episode,
        step_idx,
        "retreat",
        release_target,
        retreat_target,
        int(args.retreat_steps),
        grip_alpha=0.0,
        start_progress=0.0,
        target_noise=target_noise,
    )
    return _finish_episode(logger, args, success), obs, int(success and logger is not None)


def _run_smooth_segment(
    env,
    obs: dict,
    args: argparse.Namespace,
    frame: TcpToWristFrame,
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    logger,
    episode: int,
    step_idx: int,
    phase: str,
    start: torch.Tensor,
    goal: torch.Tensor,
    steps: int,
    *,
    grip_alpha: float,
    start_progress: float,
    target_noise: TargetNoiseState | None = None,
) -> tuple[dict, int, bool]:
    num_steps = max(int(steps), 1)
    for i in range(num_steps):
        target = _smooth_line_target(
            start,
            goal,
            i,
            num_steps,
            start_progress=start_progress,
        )
        if target_noise is not None:
            target = target_noise.apply(target, phase, final=i >= num_steps - 1)
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            target,
            grip_alpha=grip_alpha,
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase=phase,
        )
        _log_step(args, episode, step_idx, phase, obs, target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            return obs, step_idx, success
    return obs, step_idx, False


def _run_arc_segment(
    env,
    obs: dict,
    args: argparse.Namespace,
    frame: TcpToWristFrame,
    slices,
    low: torch.Tensor,
    high: torch.Tensor,
    logger,
    episode: int,
    step_idx: int,
    phase: str,
    start: torch.Tensor,
    goal: torch.Tensor,
    steps: int,
    *,
    grip_alpha: float,
    arc_height: float,
    start_progress: float,
    release_grip_start: float | None = None,
    target_noise: TargetNoiseState | None = None,
) -> tuple[dict, int, bool]:
    num_steps = max(int(steps), 1)
    for i in range(num_steps):
        target = _smooth_arc_target(
            start,
            goal,
            i,
            num_steps,
            arc_height=arc_height,
            start_progress=start_progress,
        )
        if target_noise is not None:
            target = target_noise.apply(target, phase, final=i >= num_steps - 1)
        grip = grip_alpha
        if release_grip_start is not None:
            raw = i / max(float(num_steps - 1), 1.0)
            start_release = max(0.0, min(float(release_grip_start), 0.99))
            release_u = _smoothstep((raw - start_release) / max(1.0 - start_release, 1e-6))
            grip = grip_alpha * (1.0 - release_u)
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            target,
            grip_alpha=grip,
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase=phase,
        )
        _log_step(args, episode, step_idx, phase, obs, target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            return obs, step_idx, success
    return obs, step_idx, False


def _smooth_line_target(
    start: torch.Tensor,
    goal: torch.Tensor,
    step: int,
    steps: int,
    *,
    start_progress: float,
) -> torch.Tensor:
    u = _smooth_progress(step, steps, start_progress=start_progress)
    target = (1.0 - u) * start + u * goal
    if step >= max(int(steps), 1) - 1:
        target = goal.detach().clone()
    return target.contiguous()


def _smooth_arc_target(
    start: torch.Tensor,
    goal: torch.Tensor,
    step: int,
    steps: int,
    *,
    arc_height: float,
    start_progress: float,
) -> torch.Tensor:
    u = _smooth_progress(step, steps, start_progress=start_progress)
    target = (1.0 - u) * start + u * goal
    target = target.detach().clone()
    target[2] += max(float(arc_height), 0.0) * math.sin(math.pi * u)
    if step >= max(int(steps), 1) - 1:
        target = goal.detach().clone()
    return target.contiguous()


def _smooth_progress(
    step: int,
    steps: int,
    *,
    start_progress: float,
) -> float:
    denom = max(int(steps), 1)
    t = float(step + 1) / float(denom)
    smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    start_u = max(0.0, min(float(start_progress), 0.25))
    return start_u + (1.0 - start_u) * smooth


def _cube_target(obs: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    target = _object_pos(obs, "cube_pos", device)
    target = target.detach().clone()
    target[2] += float(args.grasp_z_offset)
    return target


def _place_cube_center(
    obs: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    goal = _object_pos(obs, "goal_pos", device)
    target = goal.detach().clone()
    target[2] = goal[2] + float(args.release_cube_z_offset)
    return target


def _log_step(
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
    cube = _object_pos(obs, "cube_pos", target.device)
    goal = _object_pos(obs, "goal_pos", target.device)
    cube_goal = float(torch.linalg.norm(cube - goal))
    print(
        f"episode={episode} step={step_idx} phase={phase} "
        f"target_err={float(torch.linalg.norm(tcp - target)):.4f} "
        f"tcp_cube={_fmt(tcp - cube)} cube_goal={cube_goal:.4f} success={success}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
