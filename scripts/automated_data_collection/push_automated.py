"""Direct automated push with osc_pose_abs_quat.

The push contact is chosen from the cube orientation. The robot approaches the
rotated face opposite the goal direction, then pushes through the goal with a
single low, smooth target sequence.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch

try:
    from scripts.automated_data_collection.lift_automated import (
        TcpToWristFrame,
        _action_slices,
        _control_pos,
        _finish_episode,
        _fmt,
        _make_env,
        _object_pos,
        _smoothstep,
        _step_target,
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
        TcpToWristFrame,
        _action_slices,
        _control_pos,
        _finish_episode,
        _fmt,
        _make_env,
        _object_pos,
        _smoothstep,
        _step_target,
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

CUBE_SIZE = 0.05


@dataclass(frozen=True)
class PushGeometry:
    """Geometry targets for one push episode."""

    cube: torch.Tensor
    goal: torch.Tensor
    direction_xy: torch.Tensor
    face_normal_xy: torch.Tensor
    precontact: torch.Tensor
    contact: torch.Tensor
    push_end: torch.Tensor
    cube_yaw_rad: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the push collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="push")
    parser.add_argument("--manipulator", default="franka")
    parser.add_argument("--gripper", default="umi")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render-mode", choices=("human", "rgb_array", "none"), default="none")
    parser.add_argument("--cameras", default="front,wrist,right_side")
    parser.add_argument("--modalities", default="rgb")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--push-z-offset", type=float, default=0.0)
    parser.add_argument("--contact-margin", type=float, default=0.004)
    parser.add_argument("--approach-backoff", type=float, default=0.075)
    parser.add_argument("--push-overshoot", type=float, default=0.040)
    parser.add_argument("--approach-steps", type=int, default=80)
    parser.add_argument("--approach-arc-height", type=float, default=0.045)
    parser.add_argument("--contact-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=70)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--start-progress", type=float, default=0.02)
    parser.add_argument("--push-grip-alpha", type=float, default=0.35)
    add_target_noise_args(parser)
    add_keyboard_gripper_args(parser)
    return parser.parse_args()


def main() -> int:
    """Run direct automated push collection."""
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
            geom = _push_geometry(obs, args, env.device)
            print(
                f"episode={episode} tcp={_fmt(_tcp_pos(obs, env.device))} "
                f"cube={_fmt(geom.cube)} goal={_fmt(geom.goal)} "
                f"dir={_fmt2(geom.direction_xy)} face={_fmt2(geom.face_normal_xy)} "
                f"yaw={math.degrees(geom.cube_yaw_rad):+.1f}deg",
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


def _run_episode(
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
    target_noise = TargetNoiseState.from_args(args, episode=episode)

    geom = _push_geometry(obs, args, env.device)
    approach_start = _control_pos(env, obs, env.device).detach().clone()
    obs, step_idx, success = _run_segment(
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
        "push:approach",
        approach_start,
        geom.precontact,
        int(args.approach_steps),
        grip_alpha=float(args.push_grip_alpha),
        arc_height=float(args.approach_arc_height),
        target_noise=target_noise,
    )
    if success or step_idx >= int(args.max_steps):
        return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    geom = _push_geometry(obs, args, env.device)
    contact_start = _control_pos(env, obs, env.device).detach().clone()
    obs, step_idx, success = _run_segment(
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
        "push:contact",
        contact_start,
        geom.contact,
        int(args.contact_steps),
        grip_alpha=float(args.push_grip_alpha),
        arc_height=0.0,
        target_noise=target_noise,
    )
    if success or step_idx >= int(args.max_steps):
        return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    geom = _push_geometry(obs, args, env.device)
    push_start = _control_pos(env, obs, env.device).detach().clone()
    obs, step_idx, success = _run_segment(
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
        "push:through",
        push_start,
        geom.push_end,
        int(args.push_steps),
        grip_alpha=float(args.push_grip_alpha),
        arc_height=0.0,
        target_noise=target_noise,
    )
    if success or step_idx >= int(args.max_steps):
        return _finish_episode(logger, args, success), obs, int(success and logger is not None)

    for _ in range(int(args.hold_steps)):
        target = target_noise.apply(geom.push_end, "push:hold")
        obs, success = _step_target(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            target,
            grip_alpha=float(args.push_grip_alpha),
            trace=None,
            episode=episode,
            step_idx=step_idx,
            phase="push:hold",
        )
        _log_step(args, episode, step_idx, "push:hold", obs, target, success)
        step_idx += 1
        if success or step_idx >= int(args.max_steps):
            break

    return _finish_episode(logger, args, success), obs, int(success and logger is not None)


def _run_segment(
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
    target_noise: TargetNoiseState | None = None,
) -> tuple[dict, int, bool]:
    num_steps = max(int(steps), 1)
    for i in range(num_steps):
        target = _smooth_target(
            start,
            goal,
            i,
            num_steps,
            start_progress=float(args.start_progress),
            arc_height=arc_height,
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


def _smooth_target(
    start: torch.Tensor,
    goal: torch.Tensor,
    step: int,
    steps: int,
    *,
    start_progress: float,
    arc_height: float,
) -> torch.Tensor:
    denom = max(int(steps), 1)
    t = float(step + 1) / float(denom)
    start_u = max(0.0, min(float(start_progress), 0.25))
    u = start_u + (1.0 - start_u) * _smoothstep(t)
    target = (1.0 - u) * start + u * goal
    target = target.detach().clone()
    target[2] += max(float(arc_height), 0.0) * math.sin(math.pi * u)
    if step >= denom - 1:
        target = goal.detach().clone()
    return target.contiguous()


def _push_geometry(obs: dict, args: argparse.Namespace, device: torch.device) -> PushGeometry:
    cube = _object_pos(obs, "cube_pos", device)
    goal = _object_pos(obs, "goal_pos", device)
    direction = _normalize_xy(goal[:2] - cube[:2], fallback=(0.0, -1.0), device=device)
    face_normal = _push_face_normal(obs, direction, device)

    half = 0.5 * CUBE_SIZE
    contact_radius = half + max(float(args.contact_margin), 0.0)
    precontact_radius = half + max(float(args.approach_backoff), float(args.contact_margin))
    push_z = cube[2] + float(args.push_z_offset)

    precontact = cube.detach().clone()
    precontact[:2] = cube[:2] + face_normal * precontact_radius
    precontact[2] = push_z

    contact = cube.detach().clone()
    contact[:2] = cube[:2] + face_normal * contact_radius
    contact[2] = push_z

    push_end = goal.detach().clone()
    push_end[:2] = goal[:2] + face_normal * contact_radius + direction * float(args.push_overshoot)
    push_end[2] = push_z

    cube_yaw = _cube_yaw(obs, device)
    return PushGeometry(
        cube=cube.detach().clone(),
        goal=goal.detach().clone(),
        direction_xy=direction.detach().clone(),
        face_normal_xy=face_normal.detach().clone(),
        precontact=precontact.contiguous(),
        contact=contact.contiguous(),
        push_end=push_end.contiguous(),
        cube_yaw_rad=cube_yaw,
    )


def _push_face_normal(
    obs: dict,
    push_direction_xy: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    quat = _object_quat(obs, "cube_quat", device)
    rot = quat_to_rotmat_wxyz_torch(quat)
    local_x = _normalize_xy(rot[:2, 0], fallback=(1.0, 0.0), device=device)
    local_y = _normalize_xy(rot[:2, 1], fallback=(0.0, 1.0), device=device)
    candidates = torch.stack((local_x, -local_x, local_y, -local_y), dim=0)
    back_direction = -push_direction_xy
    best = int(torch.argmax(candidates.matmul(back_direction)).item())
    return candidates[best].contiguous()


def _cube_yaw(obs: dict, device: torch.device) -> float:
    quat = _object_quat(obs, "cube_quat", device)
    rot = quat_to_rotmat_wxyz_torch(quat)
    return float(torch.atan2(rot[1, 0], rot[0, 0]))


def _object_quat(obs: dict, key: str, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(obs["state"]["other"][key], dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[0]
    quat = tensor.reshape(-1)[:4].contiguous()
    return quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)


def _normalize_xy(
    vec: torch.Tensor,
    *,
    fallback: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(vec, dtype=torch.float32, device=device).reshape(2)
    norm = torch.linalg.norm(value)
    if float(norm) < 1e-6:
        return torch.tensor(fallback, dtype=torch.float32, device=device)
    return (value / norm).contiguous()


def _cube_goal_distance(obs: dict, device: torch.device) -> float:
    cube = _object_pos(obs, "cube_pos", device)
    goal = _object_pos(obs, "goal_pos", device)
    return float(torch.linalg.norm(cube - goal))


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
    geom = _push_geometry(obs, args, target.device)
    print(
        f"episode={episode} step={step_idx} phase={phase} "
        f"target_err={float(torch.linalg.norm(tcp - target)):.4f} "
        f"cube_goal={_cube_goal_distance(obs, target.device):.4f} "
        f"dir={_fmt2(geom.direction_xy)} face={_fmt2(geom.face_normal_xy)} "
        f"yaw={math.degrees(geom.cube_yaw_rad):+.1f}deg success={success}",
    )


def _fmt2(vec: torch.Tensor) -> str:
    values = [float(v) for v in vec.detach().cpu().reshape(-1)[:2]]
    return "[" + ",".join(f"{v:+.3f}" for v in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
