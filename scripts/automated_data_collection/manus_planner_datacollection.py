"""Run an automated planner while commanding the gripper from Manus."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence

import torch

from dexsuite.devices.manus_geort import ManusGeort

try:
    from scripts.automated_data_collection.reach_automated import ReachPlanner
    from scripts.automated_data_collection.utils.runner import (
        ArmTargetFilter,
        GripperProfile,
        TargetNoiseState,
        TcpToWristFrame,
        _action_slices,
        _capture_video_frame,
        _make_action,
        _make_env,
        _save_video,
        _smooth_arm_action,
        _target_orientation,
        _task_summary,
        _tune_args,
        make_parser,
        tcp_pos,
    )
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from reach_automated import ReachPlanner  # type: ignore[no-redef]
    from utils.runner import (  # type: ignore[no-redef]
        ArmTargetFilter,
        GripperProfile,
        TargetNoiseState,
        TcpToWristFrame,
        _action_slices,
        _capture_video_frame,
        _make_action,
        _make_env,
        _save_video,
        _smooth_arm_action,
        _target_orientation,
        _task_summary,
        _tune_args,
        make_parser,
        tcp_pos,
    )

PLANNERS = {"reach": ReachPlanner}


def _manus_gripper_action(
    manus: ManusGeort,
    *,
    width: int,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    q = torch.as_tensor(manus.get_action(), dtype=torch.float32, device=device).reshape(-1)
    if q.numel() < width:
        q = torch.cat((q, torch.zeros(width - q.numel(), dtype=torch.float32, device=device)))
    q = q[:width]
    return torch.minimum(torch.maximum(q, low), high)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Run the planner with Manus driving the gripper slice."""
    parser = make_parser(ReachPlanner)
    parser.set_defaults(manipulator="ur5", gripper="allegro")
    parser.add_argument("--manus-url", default="ws://127.0.0.1:8767")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.task not in PLANNERS:
        supported = ", ".join(sorted(PLANNERS))
        raise ValueError(f"No shared automated planner for task {args.task!r}. Supported: {supported}.")
    planner_cls = PLANNERS[args.task]
    if args.viewer:
        args.render_mode = "human"
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx" if args.render_mode == "human" else "egl")
    _tune_args(args)

    env = _make_env(args)
    logger = None
    manus = None
    try:
        obs, _info = env.reset()
        if env.horizon is not None and int(args.max_steps) > int(env.horizon):
            env.horizon = int(args.max_steps)
        slices = _action_slices(env)
        grip_width = slices.gripper.stop - slices.gripper.start
        profile = GripperProfile(env=env, gripper_name=args.gripper, task_name=args.task)
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)
        manus = ManusGeort(url=args.manus_url, expected_dim=grip_width)
        video_frames = [] if args.video_out is not None else None

        print(
            f"task={args.task} planner={planner_cls.__name__} "
            f"manipulator={args.manipulator} gripper={args.gripper} manus={args.manus_url}",
            flush=True,
        )

        if args.dry_run:
            plan = planner_cls(args, profile).plan(obs)
            print(f"action_dim={slices.total_dim} gripper_dim={grip_width}")
            print(f"first_primitive={plan.current_name}")
            return 0

        if not args.no_record:
            from dexsuite.utils.lerobot_logger import LeRobotLogger

            logger = LeRobotLogger(task=args.task, env=env, root_dir=args.dataset_root)

        max_attempts = args.max_attempts or max(args.episodes, args.episodes * 5)
        successes = attempts = dropped = saved = 0
        while successes < args.episodes and attempts < max_attempts:
            episode_idx = attempts
            if attempts > 0:
                obs, _info = env.reset()
            attempts += 1
            plan = planner_cls(args, profile).plan(obs)
            target_quat = _target_orientation(obs, args, env.device)
            control_frame = TcpToWristFrame.from_obs(obs, env.device)
            arm_filter = ArmTargetFilter(args, tcp_pos(obs).to(env.device))
            target_noise = TargetNoiseState.from_args(args, episode=episode_idx)
            prev_action = None
            print(f"episode={episode_idx} plan_start={plan.current_name}")

            for step_idx in range(args.max_steps):
                primitive_name = plan.current_name
                control_pos = profile.control_pos(obs).to(env.device)
                command = plan.command(obs, current_pos=control_pos)
                target = target_noise.apply(
                    command.target.to(env.device),
                    primitive_name,
                    final=plan.done,
                )
                action_target = arm_filter.target(
                    target,
                    control_pos,
                    control_arm=command.control_arm,
                )
                action = _make_action(
                    obs=obs,
                    target=action_target,
                    target_quat=target_quat,
                    control_frame=control_frame,
                    grip_alpha=command.grip,
                    gripper=profile,
                    slices=slices,
                    low=low,
                    high=high,
                    arm_mode=args.arm_mode,
                    pos_gain=args.pos_gain,
                    max_pos_action=args.max_pos_action,
                    control_arm=command.control_arm,
                )
                if command.control_arm:
                    action = _smooth_arm_action(action, prev_action, slices.arm, args.arm_smoothing)
                action = action.clone()
                action[slices.gripper] = _manus_gripper_action(
                    manus,
                    width=grip_width,
                    low=low[slices.gripper],
                    high=high[slices.gripper],
                    device=env.device,
                )

                with torch.no_grad():
                    next_obs, reward, terminated, truncated, info = env.step(action)
                if video_frames is not None and step_idx % max(1, args.video_every) == 0:
                    frame = _capture_video_frame(env, args.video_camera)
                    if frame is not None:
                        video_frames.append(frame)
                prev_action = action.detach().clone()

                if logger is not None:
                    logger.add_step_pre(
                        obs_before=obs,
                        action=action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )

                success = bool(torch.as_tensor(info.get("success", False)).reshape(-1)[0])
                is_done = bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(
                    torch.as_tensor(truncated).reshape(-1)[0],
                )
                if step_idx % max(1, args.print_every) == 0:
                    err = float(torch.linalg.norm(control_pos - target))
                    print(
                        f"episode={episode_idx} step={step_idx} primitive={primitive_name} "
                        f"err={err:.4f} success={success} "
                        f"reward={float(torch.as_tensor(reward).reshape(-1)[0]):+.4f}"
                        f"{_task_summary(args.task, obs)}",
                    )

                obs = next_obs
                if success:
                    successes += 1
                    if logger is not None:
                        logger.save_episode()
                        saved += 1
                    print(f"episode={episode_idx} saved success")
                    break
                if is_done or plan.done:
                    if logger is not None and args.save_all:
                        logger.save_episode()
                        saved += 1
                    elif logger is not None:
                        logger.clear_buffer()
                    dropped += 1
                    print(f"episode={episode_idx} dropped")
                    break
            else:
                if logger is not None:
                    logger.clear_buffer()
                dropped += 1
                print(f"episode={episode_idx} dropped max_steps")

        if video_frames is not None and args.video_out is not None:
            _save_video(args.video_out, video_frames, fps=float(args.control_hz) / args.video_every)
        print(f"done saved={saved} successes={successes} dropped={dropped} attempts={attempts}")
        return 0
    finally:
        if logger is not None:
            logger.close()
        if manus is not None:
            with contextlib.suppress(Exception):
                manus.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
