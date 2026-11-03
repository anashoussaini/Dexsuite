"""Environment runner for builder specs (torch-only actions).

This module intentionally keeps teleoperation support minimal and strict:
- The environment is always stepped with a torch.Tensor action.
- Device adapters return a flat action vector matching env.action_space.

Device support:
- keyboard: Uses dexsuite.devices.Keyboard when available (pynput-based).
- spacemouse: Uses dexsuite.devices.Spacemouse (requires evdev on Linux, or hidapi
  on macOS/Windows).
- vive_controller / vive_tracker: Uses Vive devices (requires openvr).
- none: Sends zero actions.

Notes:
    This runner focuses on single-arm teleop. For bimanual robots you can still
    run the environment with --input none (or extend this module to drive each
    arm independently).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

if __package__:
    from scripts.interactive_builder.spec import BuilderSpec
else:
    from spec import BuilderSpec


class ActionSource(Protocol):
    """Produce torch actions for env.step(action)."""

    def next_action(self) -> torch.Tensor:
        """Return the next flat action tensor."""
        ...

    def close(self) -> None:
        """Release any device resources held by the action source."""
        ...


def run_from_spec(spec: BuilderSpec, *, input_device: str | None = None) -> int:
    """Build an environment from spec and run a teleop loop.

    Args:
        spec: Builder spec.
        input_device: Teleoperation device name.

    Returns:
        Exit code (0 success).
    """
    import dexsuite as ds

    input_device = spec.input_device if input_device is None else str(input_device)

    if spec.type_of_robot == "bimanual" and input_device != "none":
        raise ValueError(
            "Teleop input is currently implemented for single-arm robots only. "
            "Use --input none for bimanual (or extend runner.py).",
        )

    kw = spec.to_make_kwargs()
    env = ds.make(spec.task, **kw)
    try:
        env.reset()
        src = _make_action_source(env, input_device=str(input_device))
        try:
            _loop(env, spec, src)
        finally:
            src.close()
    finally:
        env.close()
    return 0


def _loop(env, spec: BuilderSpec, src: ActionSource) -> None:
    dt = 1.0 / float(max(1, spec.control_hz))
    n_envs = int(getattr(env, "n_envs", 1) or 1)

    last = time.time()
    try:
        while True:
            a = src.next_action()
            if not torch.is_tensor(a):
                raise TypeError("ActionSource must return a torch.Tensor.")
            if a.dtype != torch.float32:
                a = a.to(dtype=torch.float32)

            if n_envs > 1 and a.ndim == 1:
                a = a.unsqueeze(0).repeat(n_envs, 1)

            with torch.no_grad():
                _obs, _reward, terminated, truncated, info = env.step(a)

            # Best-effort done handling across scalar/tensor returns. DexSuite
            # BaseEnv already auto-resets and reports that via info["needs_reset"];
            # only reset here for Gym-style envs that do not do it internally.
            done = _as_bool_any(terminated) or _as_bool_any(truncated)
            if done and "needs_reset" not in info:
                env.reset()

            # Frame pacing (best-effort)
            now = time.time()
            sleep = dt - (now - last)
            if sleep > 0:
                time.sleep(sleep)
            last = now
    except KeyboardInterrupt:
        return


def _as_bool_any(x: Any) -> bool:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return bool(x.item())
        return bool(torch.any(x).item())
    return bool(x)


def _robot_device(env) -> torch.device:
    dev = getattr(getattr(env, "robot", None), "device", None)
    if isinstance(dev, torch.device):
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class _LayoutInfo:
    arm_dim: int
    grip_dim: int


def _tensor1d_or_none(x: Any, *, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)
    return t.contiguous()


def _expand_scalar_gripper_action(
    env,
    *,
    scalar: torch.Tensor,
    grip_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a 1D open/close command into the gripper controller action space.

    Builder devices expose a single gripper scalar, but several DexSuite
    grippers use multiple joints. Repeating the same scalar across all joints is
    wrong for mirrored hands like Franka tactile, where the open pose is
    `[+d, -d]`. Instead, interpolate from the model's open target at -1 to its
    closed target at +1, then map that physical target back into the
    controller's action space.
    """
    s = scalar.to(device=device, dtype=torch.float32).reshape(())
    robot = getattr(env, "robot", None)
    hand_ctrl = getattr(robot, "hand_ctrl", None)
    gripper_model = getattr(robot, "gripper", None)

    open_q = _tensor1d_or_none(getattr(gripper_model, "open_q", None), device=device)
    if open_q is None:
        open_q = _tensor1d_or_none(
            getattr(gripper_model, "home_q", None), device=device
        )
    if open_q is None or open_q.numel() != grip_dim:
        return s.view(1).repeat(grip_dim)

    close_q = _tensor1d_or_none(
        getattr(gripper_model, "closed_q", None),
        device=device,
    )
    if close_q is None:
        close_q = torch.zeros_like(open_q)
    if close_q.numel() != grip_dim:
        return s.view(1).repeat(grip_dim)

    alpha = ((s.clamp(-1.0, 1.0) + 1.0) * 0.5).to(dtype=torch.float32)
    q_target = torch.lerp(open_q, close_q, alpha).contiguous()

    normalized = bool(getattr(hand_ctrl, "normalized", False))
    if not normalized:
        return q_target

    lo = _tensor1d_or_none(getattr(hand_ctrl, "_lo_1d", None), device=device)
    hi = _tensor1d_or_none(getattr(hand_ctrl, "_hi_1d", None), device=device)
    if lo is None or hi is None or lo.numel() != grip_dim or hi.numel() != grip_dim:
        lo = _tensor1d_or_none(getattr(gripper_model, "q_lo", None), device=device)
        hi = _tensor1d_or_none(getattr(gripper_model, "q_hi", None), device=device)
    if lo is None or hi is None or lo.numel() != grip_dim or hi.numel() != grip_dim:
        return s.view(1).repeat(grip_dim)

    denom = hi - lo
    safe_denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
    return (2.0 * (q_target - lo) / safe_denom - 1.0).clamp(-1.0, 1.0).contiguous()


def _normalize_gripper_action(
    env,
    *,
    gripper: torch.Tensor,
    grip_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a raw device gripper action to the controller action shape."""
    grip = gripper.to(device=device, dtype=torch.float32).flatten()
    if grip.numel() == 1 and grip_dim > 1:
        grip = _expand_scalar_gripper_action(
            env,
            scalar=grip[0],
            grip_dim=grip_dim,
            device=device,
        )
    if grip.numel() != grip_dim:
        raise ValueError(
            f"gripper action dim mismatch: expected {grip_dim}, got {grip.numel()}",
        )
    return grip.contiguous()


def _current_gripper_action(
    env,
    *,
    grip_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Map the gripper's current physical joint state back into action space."""
    robot = getattr(env, "robot", None)
    hand_ctrl = getattr(robot, "hand_ctrl", None)
    hand_ent = getattr(robot, "hand_ent", None)
    if hand_ctrl is None or hand_ent is None:
        return torch.zeros(grip_dim, dtype=torch.float32, device=device)

    qpos = hand_ent.get_dofs_position()
    qpos = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    qpos_1d = qpos[0] if qpos.ndim == 2 else qpos
    qpos_1d = qpos_1d.reshape(-1)

    dofs_idx = getattr(hand_ctrl, "dofs_idx", None)
    ctrl_q = qpos_1d[dofs_idx] if dofs_idx is not None else qpos_1d[:grip_dim]

    if ctrl_q.numel() != grip_dim:
        return torch.zeros(grip_dim, dtype=torch.float32, device=device)

    normalized = bool(getattr(hand_ctrl, "normalized", False))
    if not normalized:
        return ctrl_q.contiguous()

    lo = _tensor1d_or_none(getattr(hand_ctrl, "_lo_1d", None), device=device)
    hi = _tensor1d_or_none(getattr(hand_ctrl, "_hi_1d", None), device=device)
    if lo is None or hi is None or lo.numel() != grip_dim or hi.numel() != grip_dim:
        return torch.zeros(grip_dim, dtype=torch.float32, device=device)

    denom = hi - lo
    safe_denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
    return (2.0 * (ctrl_q - lo) / safe_denom - 1.0).clamp(-1.0, 1.0).contiguous()


def _layout_info(env) -> _LayoutInfo:
    robot = getattr(env, "robot", None)
    layout = getattr(robot, "_layout", None)
    if layout is None or not getattr(layout, "segments", None):
        raise RuntimeError("Robot action layout not available (robot._layout missing).")

    def _w(seg) -> int:
        return int(seg.stop - seg.start)

    segs = list(layout.segments)
    # Single-arm convention: [manipulator, gripper?]
    arm_dim = _w(segs[0]) if segs else 0
    grip_dim = _w(segs[1]) if len(segs) > 1 else 0
    return _LayoutInfo(arm_dim=arm_dim, grip_dim=grip_dim)


def _pack_single_action(
    env,
    *,
    pose: torch.Tensor | None,
    gripper: torch.Tensor | None,
) -> torch.Tensor:
    info = _layout_info(env)
    dev = _robot_device(env)
    if pose is None:
        pose = torch.zeros(info.arm_dim, dtype=torch.float32, device=dev)
    pose = pose.to(device=dev, dtype=torch.float32).flatten()
    if pose.numel() != info.arm_dim:
        raise ValueError(
            f"arm action dim mismatch: expected {info.arm_dim}, got {pose.numel()}",
        )

    if info.grip_dim == 0:
        return pose.contiguous()

    if gripper is None:
        grip = torch.zeros(info.grip_dim, dtype=torch.float32, device=dev)
    else:
        grip = _normalize_gripper_action(
            env,
            gripper=gripper,
            grip_dim=info.grip_dim,
            device=dev,
        )
    return torch.cat([pose, grip], dim=0).contiguous()


class _ZeroSource:
    def __init__(self, env) -> None:
        self._env = env
        self._dev = _robot_device(env)
        self._dim = int(getattr(env.action_space, "shape", (0,))[0])

    def next_action(self) -> torch.Tensor:
        return torch.zeros(self._dim, dtype=torch.float32, device=self._dev)

    def close(self) -> None:
        return None


class _DexsuiteDictDeviceSource:
    """Wrap a dexsuite.devices device instance and pack its dict actions into a tensor."""

    def __init__(self, env, *, device_obj) -> None:
        self._env = env
        self._dev_obj = device_obj
        self._dev = _robot_device(env)
        info = _layout_info(env)
        self._grip_dim = info.grip_dim
        self._grip_hold: torch.Tensor | None = None

    def _resolve_gripper(self, raw: Any) -> torch.Tensor | None:
        if self._grip_dim == 0:
            return None
        if self._grip_hold is None:
            self._grip_hold = _current_gripper_action(
                self._env,
                grip_dim=self._grip_dim,
                device=self._dev,
            )
        if raw is None:
            return self._grip_hold

        grip = torch.as_tensor(raw, dtype=torch.float32, device=self._dev).flatten()
        if grip.numel() == 0 or bool(torch.all(grip.abs() <= 1e-6).item()):
            return self._grip_hold

        resolved = _normalize_gripper_action(
            self._env,
            gripper=grip,
            grip_dim=self._grip_dim,
            device=self._dev,
        )
        self._grip_hold = resolved.clone()
        return resolved

    def next_action(self) -> torch.Tensor:
        d = self._dev_obj.get_action()
        pose = d.get("pose")
        grip = self._resolve_gripper(d.get("gripper"))
        pose_t = None if pose is None else torch.as_tensor(pose, dtype=torch.float32)
        grip_t = None if grip is None else torch.as_tensor(grip, dtype=torch.float32)
        return _pack_single_action(self._env, pose=pose_t, gripper=grip_t)

    def close(self) -> None:
        if hasattr(self._dev_obj, "close"):
            with contextlib.suppress(Exception):
                self._dev_obj.close()


def _require_arm_dim(dims: _LayoutInfo, *, input_device: str, expected: int) -> None:
    if dims.arm_dim != expected:
        raise ValueError(
            f"{input_device} teleop currently supports only {expected}D pose "
            f"controllers (got arm_dim={dims.arm_dim}).",
        )


def _make_keyboard_source(env, dims: _LayoutInfo) -> ActionSource:
    _require_arm_dim(dims, input_device="keyboard", expected=6)
    try:
        from dexsuite.devices import Keyboard  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Keyboard teleop requires 'pynput' (pip install pynput).",
        ) from exc

    dev = Keyboard(
        controller="OSCPose",
        normalized=True,
        gripper=(dims.grip_dim > 0),
    )
    return _DexsuiteDictDeviceSource(env, device_obj=dev)


def _make_spacemouse_source(env, dims: _LayoutInfo) -> ActionSource:
    _require_arm_dim(dims, input_device="spacemouse", expected=6)
    from dexsuite.devices import Spacemouse  # type: ignore

    try:
        dev = Spacemouse(
            path_or_ids=None,
            controller="OSCPose",
            normalized=True,
            gripper=(dims.grip_dim > 0),
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Spacemouse teleop requires dexsuite[spacemouse] and OS access to the device "
            "(/dev/input on Linux, or HIDAPI on macOS/Windows).",
        ) from exc
    return _DexsuiteDictDeviceSource(env, device_obj=dev)


def _make_vive_controller_source(env, dims: _LayoutInfo) -> ActionSource:
    _require_arm_dim(dims, input_device="vive_controller", expected=7)
    try:
        from dexsuite.devices import ViveController  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Vive controller teleop requires 'openvr'.") from exc
    return _DexsuiteDictDeviceSource(env, device_obj=ViveController(side="left"))


def _make_vive_tracker_source(env, dims: _LayoutInfo) -> ActionSource:
    _require_arm_dim(dims, input_device="vive_tracker", expected=7)
    try:
        from dexsuite.devices import ViveTracker  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Vive tracker teleop requires 'openvr'.") from exc
    return _DexsuiteDictDeviceSource(env, device_obj=ViveTracker(side="center"))


def _make_action_source(env, *, input_device: str) -> ActionSource:
    input_device = str(input_device).lower()
    if input_device == "none":
        return _ZeroSource(env)

    dims = _layout_info(env)
    if input_device == "keyboard":
        return _make_keyboard_source(env, dims)
    if input_device == "spacemouse":
        return _make_spacemouse_source(env, dims)
    if input_device == "vive_controller":
        return _make_vive_controller_source(env, dims)
    if input_device == "vive_tracker":
        return _make_vive_tracker_source(env, dims)

    raise ValueError(f"Unknown input_device: {input_device!r}")


if __name__ == "__main__":
    if __package__:
        from scripts.interactive_builder.cli import main
    else:
        from cli import main

    raise SystemExit(main())
