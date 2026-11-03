"""Reusable planning helpers for automated data collection."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from roboticstoolbox import mstraj

VecTarget = Callable[[dict], torch.Tensor]
ReachedTarget = Callable[[dict, torch.Tensor, torch.Tensor], bool]


@dataclass(frozen=True)
class PlanCommand:
    """One command emitted by a planner."""

    target: torch.Tensor
    grip: float
    control_arm: bool
    goal: torch.Tensor


@dataclass(frozen=True)
class Primitive:
    """One smooth closed-loop primitive."""

    name: str
    target: VecTarget
    steps: int
    grip_start: float
    grip_end: float
    tolerance: float = 0.025
    min_steps: int = 4
    advance_on_timeout: bool = True
    via_target: VecTarget | None = None
    grip_phase_start: float = 0.0
    grip_phase_end: float = 1.0
    control_arm: bool = True
    xy_tolerance: float | None = None
    z_tolerance: float | None = None
    reached: ReachedTarget | None = None

    def grip(self, step_idx: int) -> float:
        """Return the scalar gripper command for the primitive-local step."""
        raw = step_idx / max(float(self.steps - 1), 1.0)
        start = max(0.0, min(1.0, self.grip_phase_start))
        end = max(start + 1e-6, min(1.0, self.grip_phase_end))
        frac = smoothstep((raw - start) / (end - start))
        return self.grip_start + (self.grip_end - self.grip_start) * frac


class PrimitivePlan:
    """Executes primitives sampled by Robotics Toolbox trajectories."""

    def __init__(
        self,
        primitives: Sequence[Primitive],
        *,
        control_hz: float,
        blend_time: float,
    ) -> None:
        """Create a plan from ordered primitives."""
        if not primitives:
            raise ValueError("Automated collection plan is empty.")
        self._primitives = list(primitives)
        self._dt = 1.0 / max(float(control_hz), 1e-6)
        self._blend_time = max(0.0, float(blend_time))
        self._idx = 0
        self._local_step = 0
        self._rtb_samples: torch.Tensor | None = None
        self._rtb_goal: torch.Tensor | None = None
        self._last_target: torch.Tensor | None = None

    @property
    def done(self) -> bool:
        """Whether all primitives have completed."""
        return self._idx >= len(self._primitives)

    @property
    def current_name(self) -> str:
        """Name of the active primitive."""
        if self.done:
            return "done"
        return self._primitives[self._idx].name

    def command(
        self,
        obs: dict,
        current_pos: torch.Tensor | None = None,
    ) -> PlanCommand:
        """Return the current TCP target and scalar grip command."""
        current_pos = tcp_pos(obs) if current_pos is None else current_pos
        if self.done:
            return PlanCommand(
                target=current_pos,
                grip=0.0,
                control_arm=True,
                goal=current_pos,
            )

        prim = self._primitives[self._idx]
        if not prim.control_arm:
            target = (
                current_pos.detach().clone()
                if self._last_target is None
                else self._last_target.detach().clone()
            )
            grip = prim.grip(self._local_step)
            self._local_step += 1
            if self._local_step >= prim.steps:
                self._advance()
            self._last_target = target.detach().clone()
            return PlanCommand(
                target=target,
                grip=grip,
                control_arm=False,
                goal=target,
            )

        if self._rtb_samples is None or self._rtb_goal is None:
            start = current_pos if self._last_target is None else self._last_target
            via = prim.via_target(obs).detach().clone() if prim.via_target else None
            goal = prim.target(obs).detach().clone()
            self._rtb_goal = goal
            self._rtb_samples = rtb_xyz_samples(
                start=start.detach().clone(),
                via=via,
                goal=goal,
                steps=prim.steps,
                dt=self._dt,
                blend_time=self._blend_time,
            )

        sample_idx = min(self._local_step, self._rtb_samples.shape[0] - 1)
        target = self._rtb_samples[sample_idx]
        grip = prim.grip(self._local_step)
        self._local_step += 1
        if prim.reached is None:
            reached = primitive_reached(
                current_pos,
                self._rtb_goal,
                tolerance=prim.tolerance,
                xy_tolerance=prim.xy_tolerance,
                z_tolerance=prim.z_tolerance,
            )
        else:
            reached = prim.reached(obs, current_pos, self._rtb_goal)
        old_enough = self._local_step >= prim.min_steps
        timed_out = prim.advance_on_timeout and self._local_step >= prim.steps
        completed = timed_out or (old_enough and reached)
        goal = self._rtb_goal
        if completed:
            self._advance()
        self._last_target = current_pos.detach().clone() if completed else target.detach().clone()
        return PlanCommand(target=target, grip=grip, control_arm=True, goal=goal)

    def _advance(self) -> None:
        self._idx += 1
        self._local_step = 0
        self._rtb_samples = None
        self._rtb_goal = None


@dataclass(frozen=True)
class ObjectTarget:
    """Target that reads one object position from obs['state']['other']."""

    key: str

    def __call__(self, obs: dict) -> torch.Tensor:
        """Return the object's world position."""
        return object_pos(obs, self.key)


@dataclass(frozen=True)
class OffsetTarget:
    """Target that adds a fixed XYZ offset to another target."""

    target: VecTarget
    xyz: Sequence[float]

    def __call__(self, obs: dict) -> torch.Tensor:
        """Return the shifted target position."""
        base = self.target(obs)
        offset = torch.tensor(tuple(self.xyz), dtype=base.dtype, device=base.device)
        return base + offset


class TransportTarget:
    """Target helper that places a held object center at a destination."""

    def __init__(
        self,
        held_key: str,
        destination: VecTarget,
        held_center_offset: Sequence[float],
        control_pos: VecTarget,
    ) -> None:
        """Create a target for transporting one held object."""
        self._held_key = held_key
        self._destination = destination
        self._held_center_offset = tuple(float(v) for v in held_center_offset)
        self._control_pos = control_pos
        self._control_from_held: torch.Tensor | None = None

    def __call__(self, obs: dict) -> torch.Tensor:
        """Return a TCP target that moves the held object to the destination."""
        held = object_pos(obs, self._held_key)
        if self._control_from_held is None:
            self._control_from_held = (self._control_pos(obs) - held).detach().clone()
        destination = self._destination(obs)
        offset = torch.tensor(
            self._held_center_offset,
            dtype=destination.dtype,
            device=destination.device,
        )
        return destination + offset + self._control_from_held


class Planner:
    """Base class for automated data-collection planners."""

    task = ""

    def __init__(self, args: argparse.Namespace, profile: object | None = None) -> None:
        """Create a planner from parsed CLI args and gripper profile."""
        self.args = args
        self.profile = profile

    def plan(self, obs: dict) -> PrimitivePlan:
        """Build a sampled primitive plan for the current episode."""
        return PrimitivePlan(
            self.build_primitives(obs),
            control_hz=self.args.control_hz,
            blend_time=self.args.rtb_blend_time,
        )

    def build_primitives(self, obs: dict) -> Sequence[Primitive]:
        """Return the ordered primitives for this planner."""
        raise NotImplementedError

    @property
    def control_pos(self) -> VecTarget:
        """Position used by the plan for closed-loop control."""
        if self.profile is not None and hasattr(self.profile, "control_pos"):
            return self.profile.control_pos
        return tcp_pos

    def tcp_target_for_control_target(
        self,
        obs: dict,
        control_target: torch.Tensor,
    ) -> torch.Tensor:
        """Return a TCP target for the planner's controlled point target."""
        if self.profile is not None and hasattr(
            self.profile,
            "tcp_target_for_control_target",
        ):
            return self.profile.tcp_target_for_control_target(obs, control_target)
        return control_target

    def contact_target(self, target: VecTarget) -> VecTarget:
        """Return a contact target in the TCP frame used by the planner."""
        return target


def smoothstep(x: float) -> float:
    """Return a clamped smoothstep blend."""
    x = float(max(0.0, min(1.0, x)))
    return x * x * (3.0 - 2.0 * x)


def primitive_reached(
    current: torch.Tensor,
    goal: torch.Tensor,
    *,
    tolerance: float,
    xy_tolerance: float | None,
    z_tolerance: float | None,
) -> bool:
    """Return whether a primitive target has been reached."""
    if xy_tolerance is None and z_tolerance is None:
        return bool(torch.linalg.norm(current - goal) <= tolerance)
    delta = current - goal
    xy_ok = True
    z_ok = True
    if xy_tolerance is not None:
        xy_ok = bool(torch.linalg.norm(delta[:2]) <= float(xy_tolerance))
    if z_tolerance is not None:
        z_ok = bool(torch.abs(delta[2]) <= float(z_tolerance))
    return xy_ok and z_ok


def rtb_xyz_samples(
    start: torch.Tensor,
    *,
    via: torch.Tensor | None,
    goal: torch.Tensor,
    steps: int,
    dt: float,
    blend_time: float,
) -> torch.Tensor:
    """Return XYZ samples generated by Robotics Toolbox ``mstraj``."""
    device = start.device
    dtype = start.dtype
    total_steps = max(2, int(steps))
    total_time = total_steps * float(dt)
    points = [goal] if via is None else [via, goal]
    viapoints = np.vstack([p.detach().cpu().numpy() for p in points])
    traj = mstraj(
        viapoints,
        dt=float(dt),
        tacc=min(float(blend_time), max(total_time * 0.25, float(dt))),
        q0=start.detach().cpu().numpy(),
        tsegment=segment_times(start, points, total_time),
    )
    samples = torch.as_tensor(traj.q, dtype=dtype, device=device)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 3)
    if samples.shape[0] < total_steps:
        pad = samples[-1:].expand(total_steps - samples.shape[0], -1)
        samples = torch.cat((samples, pad), dim=0)
    return samples[:total_steps].contiguous()


def segment_times(
    start: torch.Tensor,
    points: Sequence[torch.Tensor],
    total_time: float,
) -> list[float]:
    """Split total motion time over segments by distance."""
    positions = [start, *points]
    dists = [
        float(torch.linalg.norm(positions[i + 1] - positions[i]))
        for i in range(len(positions) - 1)
    ]
    if len(dists) == 1:
        return [float(total_time)]
    total_dist = sum(dists)
    if total_dist <= 1e-9:
        weights = [1.0 / len(dists)] * len(dists)
    else:
        raw = [dist / total_dist for dist in dists]
        weights = [max(0.20, value) for value in raw]
        scale = sum(weights)
        weights = [value / scale for value in weights]
    return [max(float(total_time) * weight, 1e-6) for weight in weights]


def vec3(value: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Return a flat XYZ tensor from a batched or unbatched value."""
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[0]
    return tensor.reshape(-1)[:3].contiguous()


def tcp_pos(obs: dict) -> torch.Tensor:
    """Return the observed TCP position."""
    return vec3(obs["state"]["gripper"]["tcp_pos"])


def tcp_quat(obs: dict, device: torch.device | None = None) -> torch.Tensor:
    """Return the observed TCP quaternion in WXYZ convention."""
    tensor = torch.as_tensor(
        obs["state"]["gripper"]["tcp_quat"],
        dtype=torch.float32,
        device=device,
    )
    if tensor.ndim == 2:
        tensor = tensor[0]
    return quat_normalize_wxyz(tensor.reshape(-1)[:4].contiguous())


def object_pos(obs: dict, key: str) -> torch.Tensor:
    """Return one object position from obs['state']['other']."""
    return vec3(obs["state"]["other"][key])


def quat_normalize_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Normalize a WXYZ quaternion."""
    return quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)


def quat_conj_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a WXYZ quaternion."""
    return torch.stack((quat[0], -quat[1], -quat[2], -quat[3]))


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two WXYZ quaternions."""
    aw, ax, ay, az = a.unbind()
    bw, bx, by, bz = b.unbind()
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
    )


def rpy_quat_wxyz(
    roll: float,
    pitch: float,
    yaw: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a WXYZ quaternion from roll, pitch, yaw radians."""
    half_roll = torch.tensor(float(roll) * 0.5, dtype=torch.float32, device=device)
    half_pitch = torch.tensor(float(pitch) * 0.5, dtype=torch.float32, device=device)
    half_yaw = torch.tensor(float(yaw) * 0.5, dtype=torch.float32, device=device)
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
    )


def xyz_error(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return total, XY, and Z absolute errors."""
    delta = a - b
    return (
        float(torch.linalg.norm(delta)),
        float(torch.linalg.norm(delta[:2])),
        float(torch.abs(delta[2])),
    )
