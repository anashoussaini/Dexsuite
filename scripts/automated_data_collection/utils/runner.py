"""Shared DexSuite runner for automated data-collection planners."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.automated_data_collection.planner import (
        Planner,
        object_pos,
        quat_conj_wxyz,
        quat_mul_wxyz,
        quat_normalize_wxyz,
        rpy_quat_wxyz,
        tcp_pos,
        tcp_quat,
        vec3,
    )
    from scripts.automated_data_collection.utils.keyboard_gripper import (
        KeyboardGripperControl,
        add_keyboard_gripper_args,
    )
    from scripts.automated_data_collection.utils.target_noise import (
        TargetNoiseState,
        add_target_noise_args,
    )
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from planner import (  # type: ignore[no-redef]
        Planner,
        object_pos,
        quat_conj_wxyz,
        quat_mul_wxyz,
        quat_normalize_wxyz,
        rpy_quat_wxyz,
        tcp_pos,
        tcp_quat,
        vec3,
    )
    from utils.keyboard_gripper import (  # type: ignore[no-redef]
        KeyboardGripperControl,
        add_keyboard_gripper_args,
    )
    from utils.target_noise import (  # type: ignore[no-redef]
        TargetNoiseState,
        add_target_noise_args,
    )
from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch

UMI_OPEN_Q = (0.040, -0.040)
UMI_CUBE_CLOSE_Q = (0.0, 0.0)


@dataclass(frozen=True)
class ActionSlices:
    """Slices for the flat DexSuite action vector."""

    total_dim: int
    arm: slice
    gripper: slice


class ArmTargetFilter:
    """Smooth the commanded TCP target before it is sent to the pose controller."""

    def __init__(self, args: argparse.Namespace, initial_pos: torch.Tensor) -> None:
        """Create a per-episode target filter."""
        self.args = args
        self.pos = initial_pos.detach().clone()

    def target(
        self,
        planned_target: torch.Tensor,
        current_pos: torch.Tensor,
        *,
        control_arm: bool,
    ) -> torch.Tensor:
        """Return the target to send to the arm controller."""
        if not control_arm:
            self.pos = planned_target.detach().clone()
            return planned_target
        if self.args.arm_mode != "smooth":
            self.pos = planned_target.detach().clone()
            return planned_target

        error = planned_target - current_pos
        desired = planned_target + float(self.args.smooth_error_lead) * error
        delta = desired - self.pos
        dist = torch.linalg.norm(delta)
        max_step = float(self.args.smooth_max_step)
        if bool(dist > max_step):
            delta = delta * (max_step / torch.clamp(dist, min=1e-6))
        alpha = float(max(0.0, min(1.0, self.args.smooth_alpha)))
        self.pos = self.pos + alpha * delta
        return self.pos.detach().clone()


@dataclass(frozen=True)
class TcpToWristFrame:
    """Fixed episode-local transform used to command wrist targets from TCP targets."""

    wrist_origin_pos: torch.Tensor
    wrist_origin_quat: torch.Tensor
    tcp_from_wrist_pos: torch.Tensor
    tcp_from_wrist_quat: torch.Tensor

    @classmethod
    def from_obs(cls, obs: dict, device: torch.device) -> TcpToWristFrame:
        """Build the wrist-to-TCP transform from the reset observation."""
        wrist_pos = _eef_pos(obs, device)
        wrist_quat = _eef_quat(obs, device)
        tcp_world_pos = tcp_pos(obs).to(device)
        tcp_world_quat = tcp_quat(obs, device)
        wrist_rot = quat_to_rotmat_wxyz_torch(wrist_quat)
        tcp_from_wrist_pos = (
            wrist_rot.transpose(-1, -2)
            .matmul(
                (tcp_world_pos - wrist_pos).unsqueeze(-1),
            )
            .squeeze(-1)
        )
        tcp_from_wrist_quat = quat_normalize_wxyz(
            quat_mul_wxyz(quat_conj_wxyz(wrist_quat), tcp_world_quat),
        )
        return cls(
            wrist_origin_pos=wrist_pos.detach().clone(),
            wrist_origin_quat=wrist_quat.detach().clone(),
            tcp_from_wrist_pos=tcp_from_wrist_pos.detach().clone(),
            tcp_from_wrist_quat=tcp_from_wrist_quat.detach().clone(),
        )

    def wrist_target(
        self,
        tcp_target_pos: torch.Tensor,
        tcp_target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the wrist pose that places the TCP at the requested pose."""
        wrist_quat = quat_normalize_wxyz(
            quat_mul_wxyz(
                quat_normalize_wxyz(tcp_target_quat),
                quat_conj_wxyz(self.tcp_from_wrist_quat),
            ),
        )
        wrist_rot = quat_to_rotmat_wxyz_torch(wrist_quat)
        wrist_pos = tcp_target_pos - wrist_rot.matmul(
            self.tcp_from_wrist_pos.unsqueeze(-1),
        ).squeeze(-1)
        return wrist_pos, wrist_quat


class GripperProfile:
    """Map scalar open-close commands to a hand action vector."""

    def __init__(self, *, env, gripper_name: str, task_name: str) -> None:
        """Create a gripper profile from the initialized DexSuite env."""
        self.env = env
        self.device = env.device
        self.hand_ctrl = env.robot.hand_ctrl
        self.name = gripper_name.lower()
        self.task = task_name.lower()
        self.dim = int(self.hand_ctrl.action_space().shape[0])
        self.open_q = _open_q(env.robot.gripper, self.dim, self.device)
        self.close_q = _closed_q(env.robot.gripper, self.dim, self.device)
        self._control_links = _control_links(env.robot.gripper, env.robot.hand_ent)
        if self.name in ("umi", "franka_umi_tacticle") and self.dim == 2:
            self.open_q = torch.tensor(
                UMI_OPEN_Q, dtype=torch.float32, device=self.device
            )
            self.close_q = torch.tensor(
                UMI_CUBE_CLOSE_Q,
                dtype=torch.float32,
                device=self.device,
            )

    @property
    def is_umi(self) -> bool:
        """Whether this profile is the UMI parallel gripper."""
        return self.name in ("umi", "franka_umi_tacticle") and self.dim == 2

    def action(self, alpha: float) -> torch.Tensor:
        """Return a gripper action in the controller action convention."""
        a = float(max(0.0, min(1.0, alpha)))
        q = torch.lerp(self.open_q, self.close_q, a)
        return _joint_position_action(self.hand_ctrl, q)

    def control_pos(self, obs: dict) -> torch.Tensor:
        """Return the observed TCP position used for arm control."""
        if self._control_links:
            positions = [
                vec3(link.get_pos(), self.device) for link in self._control_links
            ]
            return torch.stack(positions, dim=0).mean(dim=0)
        return tcp_pos(obs)

    def tcp_target_for_control_target(
        self,
        obs: dict,
        control_target: torch.Tensor,
    ) -> torch.Tensor:
        """Return the TCP target that places the controlled point at a target."""
        if not self._control_links:
            return control_target
        current_tcp = tcp_pos(obs).to(control_target.device)
        current_control = self.control_pos(obs).to(control_target.device)
        return control_target + current_tcp - current_control

    def capture_detected(self, obs: dict) -> bool:
        """Return whether the gripper state indicates a cube between UMI pads."""
        if not self.is_umi:
            return True
        qpos = _gripper_qpos(obs)
        if qpos.numel() < 2:
            return False
        opening_each_side = min(float(qpos[0]), float(-qpos[1]))
        return 0.007 < opening_each_side < 0.035


def make_parser(planner_cls: type[Planner]) -> argparse.ArgumentParser:
    """Create the shared automated data-collection parser."""
    parser = argparse.ArgumentParser(description=planner_cls.__doc__)
    parser.add_argument("--task", default=planner_cls.task)
    parser.add_argument("--manipulator", default="franka")
    parser.add_argument("--gripper", default="umi")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--render-mode", type=_parse_render_mode, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--visualize-tcp", action="store_true")
    parser.add_argument(
        "--cameras",
        type=_parse_cameras,
        default=_parse_cameras("front,wrist,right_side"),
    )
    parser.add_argument("--modalities", type=_parse_modalities, default=("rgb",))
    parser.add_argument("--video-out", type=Path, default=None)
    parser.add_argument("--video-camera", default=None)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument(
        "--arm-mode",
        choices=("open_loop", "closed_loop", "smooth"),
        default="closed_loop",
        help=(
            "open_loop sends Robotics Toolbox trajectory samples directly; "
            "closed_loop re-servoes each sample with --pos-gain; "
            "smooth rate-limits a filtered target."
        ),
    )
    parser.add_argument("--pos-gain", type=float, default=5.0)
    parser.add_argument("--max-pos-action", type=float, default=0.35)
    parser.add_argument("--arm-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--smooth-max-step",
        type=float,
        default=0.010,
        help="Maximum TCP target motion per control step in --arm-mode smooth.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.75,
        help="Low-pass blend for each rate-limited smooth target step.",
    )
    parser.add_argument(
        "--smooth-error-lead",
        type=float,
        default=0.20,
        help="Small tracking lead added in --arm-mode smooth.",
    )
    parser.add_argument("--rtb-blend-time", type=float, default=0.20)
    parser.add_argument("--move-steps", type=int, default=120)
    parser.add_argument("--grip-steps", type=int, default=32)
    parser.add_argument("--release-steps", type=int, default=24)
    parser.add_argument("--motion-tolerance", type=float, default=0.020)
    parser.add_argument("--grasp-height", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.260)
    parser.add_argument("--grasp-retries", type=int, default=2)
    parser.add_argument("--tool-roll-deg", type=float, default=0.0)
    parser.add_argument("--tool-pitch-deg", type=float, default=0.0)
    parser.add_argument("--tool-yaw-deg", type=float, default=0.0)
    add_target_noise_args(parser)
    parser.add_argument(
        "--gripper-mode",
        choices=("auto", "manual", "keyboard"),
        default="auto",
        help=(
            "auto follows the scripted profile; manual sends --manual-gripper-q; "
            "keyboard replaces the gripper slice from o/p input."
        ),
    )
    add_keyboard_gripper_args(parser, include_mode=False)
    parser.add_argument("--manual-gripper-q", type=float, nargs="+", default=None)
    return parser


def run_collection(  # noqa: C901, PLR0912, PLR0915
    planner_cls: type[Planner],
    argv: Sequence[str] | None = None,
) -> int:
    """Run one automated data-collection planner."""
    args = make_parser(planner_cls).parse_args(None if argv is None else list(argv))
    if args.viewer:
        args.render_mode = "human"
    os.environ.setdefault(
        "PYOPENGL_PLATFORM", "glx" if args.render_mode == "human" else "egl"
    )
    _tune_args(args)

    env = _make_env(args)
    logger = None
    keyboard_gripper = KeyboardGripperControl.from_args(args)
    try:
        keyboard_gripper.start()
        obs, _info = env.reset()
        video_frames: list[np.ndarray] | None = (
            [] if args.video_out is not None else None
        )
        if env.horizon is not None and int(args.max_steps) > int(env.horizon):
            env.horizon = int(args.max_steps)
        slices = _action_slices(env)
        profile = GripperProfile(
            env=env, gripper_name=args.gripper, task_name=args.task
        )
        low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=env.device
        )
        high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=env.device
        )
        manual_gripper = _manual_gripper_action(
            args,
            slices=slices,
            low=low,
            high=high,
            device=env.device,
        )
        if manual_gripper is not None:
            compact = ",".join(f"{float(v):.4f}" for v in manual_gripper)
            print(f"manual_gripper_action=({compact})")

        if args.dry_run:
            plan = planner_cls(args, profile).plan(obs)
            print(
                f"task={args.task} gripper={args.gripper} action_dim={slices.total_dim}"
            )
            print(f"planner={planner_cls.__name__} first_primitive={plan.current_name}")
            return 0

        if not args.no_record:
            from dexsuite.utils.lerobot_logger import LeRobotLogger

            logger = LeRobotLogger(task=args.task, env=env, root_dir=args.dataset_root)

        max_attempts = args.max_attempts or max(args.episodes, args.episodes * 5)
        successes = 0
        attempts = 0
        dropped = 0
        saved = 0
        while successes < args.episodes and attempts < max_attempts:
            episode_idx = attempts
            if attempts > 0:
                obs, _info = env.reset()
            attempts += 1
            planner = planner_cls(args, profile)
            plan = planner.plan(obs)
            target_quat = _target_orientation(obs, args, env.device)
            control_frame = TcpToWristFrame.from_obs(obs, env.device)
            arm_filter = ArmTargetFilter(args, planner.control_pos(obs).to(env.device))
            target_noise = TargetNoiseState.from_args(args, episode=episode_idx)
            prev_action = None
            print(
                f"episode={episode_idx} planner={planner_cls.__name__} "
                f"plan_start={plan.current_name} move_steps={args.move_steps} "
                f"grip_steps={args.grip_steps}",
            )

            for step_idx in range(args.max_steps):
                primitive_name = plan.current_name
                control_pos = planner.control_pos(obs).to(env.device)
                command = plan.command(obs, current_pos=control_pos)
                target = command.target.to(env.device)
                target = target_noise.apply(target, primitive_name, final=plan.done)
                action_target = arm_filter.target(
                    target,
                    control_pos,
                    control_arm=command.control_arm,
                )
                effective_grip_alpha = command.grip
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
                    control_current=control_pos,
                    tcp_target_for_control_target=planner.tcp_target_for_control_target,
                    arm_mode=args.arm_mode,
                    pos_gain=args.pos_gain,
                    max_pos_action=args.max_pos_action,
                    control_arm=command.control_arm,
                )
                if command.control_arm:
                    action = _smooth_arm_action(
                        action, prev_action, slices.arm, args.arm_smoothing
                    )
                if manual_gripper is not None:
                    action = action.clone()
                    action[slices.gripper] = manual_gripper
                elif keyboard_gripper.enabled:
                    action = action.clone()
                    effective_grip_alpha = keyboard_gripper.alpha_for_step(command.grip)
                    action[slices.gripper] = profile.action(effective_grip_alpha)
                with torch.no_grad():
                    next_obs, reward, terminated, truncated, info = env.step(action)
                if (
                    video_frames is not None
                    and step_idx % max(1, args.video_every) == 0
                ):
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

                success = bool(
                    torch.as_tensor(info.get("success", False)).reshape(-1)[0]
                )
                is_done = bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(
                    torch.as_tensor(truncated).reshape(-1)[0],
                )
                if step_idx % max(1, args.print_every) == 0:
                    sample_err = float(torch.linalg.norm(control_pos - target))
                    goal_err = float(
                        torch.linalg.norm(control_pos - command.goal.to(env.device))
                    )
                    print(
                        f"episode={episode_idx} step={step_idx} primitive={primitive_name} "
                        f"sample_err={sample_err:.4f} goal_err={goal_err:.4f} "
                        f"grip={effective_grip_alpha:.3f} capture={profile.capture_detected(obs)} "
                        f"success={success} reward={float(torch.as_tensor(reward).reshape(-1)[0]):+.4f}"
                        f"{_task_summary(args.task, obs)}",
                    )

                obs = next_obs
                if success:
                    successes += 1
                    if logger is not None:
                        logger.save_episode()
                        saved += 1
                        print(f"episode={episode_idx} saved success")
                    else:
                        print(f"episode={episode_idx} success")
                    break
                if is_done or plan.done:
                    if logger is not None and args.save_all:
                        logger.save_episode()
                        saved += 1
                    elif logger is not None:
                        logger.clear_buffer()
                    dropped += 1
                    print(f"episode={episode_idx} dropped terminal_or_plan_done")
                    break
            else:
                if logger is not None and args.save_all:
                    logger.save_episode()
                    saved += 1
                elif logger is not None:
                    logger.clear_buffer()
                dropped += 1
                print(f"episode={episode_idx} dropped max_steps")

        if logger is not None:
            print(
                f"done saved={saved} successes={successes} dropped={dropped} attempts={attempts} root={logger.dataset_dir}"
            )
        else:
            print(
                f"done no_record=True successes={successes} dropped={dropped} attempts={attempts}"
            )
        if video_frames is not None and args.video_out is not None:
            fps = float(args.control_hz) / float(max(1, args.video_every))
            _save_video(args.video_out, video_frames, fps=fps)
        return 0
    finally:
        if logger is not None:
            logger.close()
        keyboard_gripper.close()
        env.close()


def _parse_cameras(value: str) -> list[str] | None:
    if value.strip().lower() in {"none", "off", "false"}:
        return None
    cameras = [part.strip() for part in value.split(",") if part.strip()]
    if not cameras:
        raise argparse.ArgumentTypeError("camera list cannot be empty")
    return cameras


def _parse_modalities(value: str) -> tuple[str, ...]:
    modalities = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modalities:
        raise argparse.ArgumentTypeError("modality list cannot be empty")
    return modalities


def _parse_render_mode(value: str) -> str | None:
    value = value.strip().lower()
    if value in {"none", "off", "false"}:
        return None
    if value not in {"human", "rgb_array"}:
        raise argparse.ArgumentTypeError(
            "render mode must be human, rgb_array, or none"
        )
    return value


def _tune_args(args: argparse.Namespace) -> None:
    if args.task.lower() == "reach":
        if args.move_steps == 120:
            args.move_steps = 60
        if args.max_steps == 800:
            args.max_steps = 100
        args.motion_tolerance = min(args.motion_tolerance, 0.020)
    if args.task.lower() == "lift":
        args.move_steps = max(args.move_steps, 120)
        args.grip_steps = max(args.grip_steps, 32)
        args.lift_height = max(args.lift_height, 0.260)
        args.motion_tolerance = min(args.motion_tolerance, 0.020)
        if args.arm_mode == "smooth":
            args.move_steps = max(args.move_steps, 200)
            args.rtb_blend_time = max(args.rtb_blend_time, 0.70)
            args.max_pos_action = min(args.max_pos_action, 0.080)


def _make_env(args: argparse.Namespace):
    import dexsuite as ds

    return ds.make(
        args.task,
        robot=ds.options.RobotOptions(
            type_of_robot="single",
            visualize_tcp=bool(args.visualize_tcp),
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
                        "position_cost": 4.0,
                        "orientation_cost": 0.3,
                        "posture_cost": 0.02,
                        "damping_cost": 0.004,
                        "max_joint_step": 0.08,
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
        sim=ds.options.SimOptions(control_hz=args.control_hz, n_envs=1),
        render_mode=args.render_mode,
        cameras=args.cameras,
        modalities=args.modalities,
    )


def _open_q(model, dim: int, device: torch.device) -> torch.Tensor:
    q = getattr(model, "open_q", None)
    if q is None:
        q = getattr(model, "home_q", None)
    if q is None:
        return torch.zeros(dim, dtype=torch.float32, device=device)
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=device).reshape(-1)
    if q_tensor.numel() != dim:
        return torch.zeros(dim, dtype=torch.float32, device=device)
    return q_tensor


def _closed_q(model, dim: int, device: torch.device) -> torch.Tensor:
    q = getattr(model, "closed_q", None)
    if q is None:
        return torch.zeros(dim, dtype=torch.float32, device=device)
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=device).reshape(-1)
    if q_tensor.numel() != dim:
        return torch.zeros(dim, dtype=torch.float32, device=device)
    return q_tensor


def _control_links(model, hand_ent) -> tuple:
    link_by_name = {str(link.name): link for link in getattr(hand_ent, "links", ())}
    return tuple(
        link_by_name[name]
        for name in (
            str(item) for item in tuple(getattr(model, "tactile_links", ()) or ())
        )
        if name in link_by_name
    )


def _joint_position_action(hand_ctrl, q_target: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(hand_ctrl, "normalized", False)):
        return q_target
    lo = getattr(hand_ctrl, "_lo_1d", None)
    hi = getattr(hand_ctrl, "_hi_1d", None)
    if lo is None or hi is None:
        raise RuntimeError("joint_position limits are not initialized.")
    lo = lo.to(dtype=torch.float32, device=q_target.device)
    hi = hi.to(dtype=torch.float32, device=q_target.device)
    span = torch.clamp(hi - lo, min=1e-6)
    return (2.0 * (q_target - lo) / span - 1.0).clamp(-1.0, 1.0)


def _action_slices(env) -> ActionSlices:
    layout = getattr(getattr(env, "robot", None), "_layout", None)
    if layout is None:
        raise RuntimeError("Robot action layout is not initialized.")
    arm_slice = None
    grip_slice = None
    for seg in layout.segments:
        name = ".".join(str(p) for p in seg.name).lower()
        if arm_slice is None and ("manipulator" in name or "arm" in name):
            arm_slice = slice(seg.start, seg.stop)
        if grip_slice is None and "gripper" in name:
            grip_slice = slice(seg.start, seg.stop)
    if arm_slice is None or grip_slice is None:
        raise ValueError("Expected one manipulator segment and one gripper segment.")
    return ActionSlices(int(layout.total_dim), arm_slice, grip_slice)


def _manual_gripper_action(
    args: argparse.Namespace,
    *,
    slices: ActionSlices,
    low: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if args.gripper_mode != "manual":
        return None
    if args.manual_gripper_q is None:
        raise ValueError("--gripper-mode manual requires --manual-gripper-q.")
    width = slices.gripper.stop - slices.gripper.start
    action = torch.tensor(
        args.manual_gripper_q, dtype=torch.float32, device=device
    ).reshape(-1)
    if action.numel() == 1 and width > 1:
        action = action.expand(width).clone()
    if action.numel() != width:
        raise ValueError(
            f"--manual-gripper-q expected {width} values, got {action.numel()}."
        )
    return torch.minimum(
        torch.maximum(action, low[slices.gripper]), high[slices.gripper]
    )


def _target_orientation(
    obs: dict, args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    current = tcp_quat(obs, device)
    delta = rpy_quat_wxyz(
        math.radians(args.tool_roll_deg),
        math.radians(args.tool_pitch_deg),
        math.radians(args.tool_yaw_deg),
        device,
    )
    return quat_normalize_wxyz(quat_mul_wxyz(delta, current))


def _eef_pos(obs: dict, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(
        obs["state"]["manipulator"]["eef_pos"],
        dtype=torch.float32,
        device=device,
    )
    if tensor.ndim == 2:
        tensor = tensor[0]
    return tensor.reshape(-1)[:3].contiguous()


def _eef_quat(obs: dict, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(
        obs["state"]["manipulator"]["eef_quat"],
        dtype=torch.float32,
        device=device,
    )
    if tensor.ndim == 2:
        tensor = tensor[0]
    return quat_normalize_wxyz(tensor.reshape(-1)[:4].contiguous())


def _make_action(
    *,
    obs: dict,
    target: torch.Tensor,
    target_quat: torch.Tensor,
    control_frame: TcpToWristFrame,
    grip_alpha: float,
    gripper: GripperProfile,
    slices: ActionSlices,
    low: torch.Tensor,
    high: torch.Tensor,
    arm_mode: str,
    pos_gain: float,
    max_pos_action: float,
    control_arm: bool,
    control_current: torch.Tensor | None = None,
    tcp_target_for_control_target: Callable[[dict, torch.Tensor], torch.Tensor]
    | None = None,
) -> torch.Tensor:
    action = torch.zeros(slices.total_dim, dtype=torch.float32, device=target.device)
    arm_dim = slices.arm.stop - slices.arm.start
    if control_current is None:
        control_current = gripper.control_pos(obs).to(target.device)
    if tcp_target_for_control_target is None:
        tcp_target_for_control_target = gripper.tcp_target_for_control_target
    if arm_dim == 7:
        control_current = control_current.to(target.device)
        if arm_mode in {"closed_loop", "smooth"}:
            pos_error = target - control_current
            control_target = control_current + torch.clamp(
                pos_gain * pos_error,
                -max_pos_action,
                max_pos_action,
            )
        else:
            control_target = target
        tcp_target = tcp_target_for_control_target(obs, control_target)
        wrist_target_pos, wrist_target_quat = control_frame.wrist_target(
            tcp_target,
            target_quat,
        )
        dq = quat_mul_wxyz(
            wrist_target_quat,
            quat_conj_wxyz(control_frame.wrist_origin_quat),
        )
        arm = torch.cat(
            (
                wrist_target_pos - control_frame.wrist_origin_pos,
                quat_normalize_wxyz(dq),
            ),
            dim=0,
        )
    elif arm_dim == 6:
        arm = torch.zeros(6, dtype=torch.float32, device=target.device)
        if control_arm:
            pos_error = target - control_current.to(target.device)
            arm[:3] = torch.clamp(pos_gain * pos_error, -max_pos_action, max_pos_action)
    else:
        raise ValueError(f"Unsupported arm action width {arm_dim}.")

    action[slices.arm] = arm
    action[slices.gripper] = gripper.action(grip_alpha)
    return torch.minimum(torch.maximum(action, low), high)


def _capture_video_frame(env, camera_name: str | None) -> np.ndarray | None:
    rendered = env.render("rgb_array")
    if rendered is None:
        return None
    if isinstance(rendered, dict):
        if not rendered:
            return None
        selected = camera_name if camera_name in rendered else next(iter(rendered))
        frame = rendered[selected]
        if isinstance(frame, dict):
            frame = frame.get("rgb")
    else:
        frame = rendered
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _save_video(path: Path, frames: list[np.ndarray], *, fps: float) -> None:
    if not frames:
        print(f"video skipped path={path} reason=no_frames")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=max(float(fps), 1.0))
    print(
        f"video saved path={path} frames={len(frames)} fps={max(float(fps), 1.0):.2f}"
    )


def _smooth_arm_action(
    action: torch.Tensor,
    prev_action: torch.Tensor | None,
    arm_slice: slice,
    smoothing: float,
) -> torch.Tensor:
    alpha = float(max(0.0, min(0.95, smoothing)))
    if alpha <= 0.0 or prev_action is None:
        return action
    action = action.clone()
    action[arm_slice] = (
        alpha * prev_action[arm_slice] + (1.0 - alpha) * action[arm_slice]
    )
    return action


def _task_summary(task: str, obs: dict) -> str:
    if task.lower() != "lift":
        return ""
    other = obs.get("state", {}).get("other", {})
    if not {"cube_pos", "target_pos"}.issubset(other):
        return ""
    cube = object_pos(obs, "cube_pos")
    target = object_pos(obs, "target_pos")
    tcp = tcp_pos(obs)
    qpos = _gripper_qpos(obs)
    xy = float(torch.linalg.norm(cube[:2] - target[:2]))
    tcp_xy = float(torch.linalg.norm(tcp[:2] - cube[:2]))
    tcp_z = float(tcp[2] - cube[2])
    q0 = float(qpos[0]) if qpos.numel() > 0 else 0.0
    q1 = float(qpos[1]) if qpos.numel() > 1 else 0.0
    return (
        f" lift=(cube_z:{float(cube[2]):.3f},cube_xy:{xy:.3f},"
        f"tcp_cube_xy:{tcp_xy:.3f},tcp_cube_z:{tcp_z:+.3f},"
        f"grip_q:[{q0:+.3f},{q1:+.3f}])"
    )


def _gripper_qpos(obs: dict) -> torch.Tensor:
    qpos = torch.as_tensor(obs["state"]["gripper"]["qpos"], dtype=torch.float32)
    if qpos.ndim == 2:
        qpos = qpos[0]
    return qpos.reshape(-1).contiguous()
