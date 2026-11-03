"""Materialize validated public `ds.make` options."""

from __future__ import annotations

from typing import Any

from .camera import CamerasOptions
from .camera_builder import build_camera_options
from .direct import build_robot_options, build_sim_options, load_controller_specs
from .layout import LayoutOptions
from .robot import ArmOptions, ControllerOptions, RobotOptions
from .sim import SimOptions
from .validation import normalize_make_options

__all__ = ["_build_robot_options", "build_make_options", "normalize_make_options"]


def build_make_options(
    kw_in: dict[str, Any],
) -> tuple[RobotOptions, CamerasOptions, SimOptions, str | None]:
    """Return resolved options for environment construction."""
    kw = dict(kw_in)
    robot_in = kw.get("robot")
    cameras_in = kw.get("cameras")
    layout_in = kw.get("layout")
    sim_in = kw.get("sim")

    robot = build_robot_options(kw, robot_in)
    if isinstance(layout_in, LayoutOptions):
        robot.layout = layout_in
        robot.__post_init__()
    elif isinstance(layout_in, str):
        if robot.type_of_robot == "single":
            raise ValueError(
                "Single-arm robots do not use layout presets. "
                "Use layout=LayoutOptions(single=PoseOptions(...)) or set "
                "robot.layout.single.",
            )
        robot.layout = LayoutOptions(preset=layout_in)
        robot.__post_init__()

    sim = build_sim_options(kw, sim_in)
    control_dt = 1.0 / float(sim.control_hz)
    robot = _materialize_robot_controllers(robot, control_dt=control_dt)

    cameras = build_camera_options(
        kw,
        cameras_in if isinstance(cameras_in, CamerasOptions) else None,
    )
    return robot, cameras, sim, kw.get("render_mode")


_ADAPTIVE_CONTROLLER_DT_KEYS: dict[str, str] = {
    "osc_pose": "mink_dt",
    "osc_pose_abs": "mink_dt",
    "osc_pose_abs_quat": "mink_dt",
    "ik_pose": "ik_dt",
}


def _materialize_controller_config(
    ctrl: ControllerOptions,
    *,
    control_dt: float,
) -> ControllerOptions:
    specs = load_controller_specs()
    yaml_defaults = dict(specs.get(ctrl.name, {}))
    user_cfg = dict(ctrl.config or {})
    merged = {**yaml_defaults, **user_cfg}
    dt_key = _ADAPTIVE_CONTROLLER_DT_KEYS.get(ctrl.name)
    if dt_key is not None and dt_key not in user_cfg:
        merged[dt_key] = float(control_dt)
    return ControllerOptions(name=ctrl.name, config=merged)


def _materialize_arm_controllers(arm: ArmOptions, *, control_dt: float) -> None:
    if isinstance(arm.manipulator_controller, ControllerOptions):
        arm.manipulator_controller = _materialize_controller_config(
            arm.manipulator_controller,
            control_dt=control_dt,
        )
    if isinstance(arm.gripper_controller, ControllerOptions):
        arm.gripper_controller = _materialize_controller_config(
            arm.gripper_controller,
            control_dt=control_dt,
        )


def _materialize_robot_controllers(
    robot_options: RobotOptions,
    *,
    control_dt: float,
) -> RobotOptions:
    if robot_options.type_of_robot == "single" and isinstance(
        robot_options.single,
        ArmOptions,
    ):
        _materialize_arm_controllers(robot_options.single, control_dt=control_dt)
        return robot_options

    if robot_options.type_of_robot == "bimanual":
        if isinstance(robot_options.left, ArmOptions):
            _materialize_arm_controllers(robot_options.left, control_dt=control_dt)
        if isinstance(robot_options.right, ArmOptions):
            _materialize_arm_controllers(robot_options.right, control_dt=control_dt)
    return robot_options


_build_robot_options = build_robot_options
