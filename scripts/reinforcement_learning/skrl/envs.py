"""Environment construction helpers for DexSuite SKRL training."""

from __future__ import annotations

from typing import Any

import dexsuite as ds

try:
    from scripts.reinforcement_learning.skrl.configs import ENV_CONFIGS
    from scripts.reinforcement_learning.skrl.wrappers import (
        DexsuiteLiftWrapper,
        DexsuiteReachWrapper,
    )
except ModuleNotFoundError:
    from configs import ENV_CONFIGS
    from wrappers import DexsuiteLiftWrapper, DexsuiteReachWrapper


WRAPPERS = {
    "reach": DexsuiteReachWrapper,
    "lift": DexsuiteLiftWrapper,
}


def _reach_make_kwargs(
    *,
    n_envs: int,
    render_mode: str | None,
    robot_preset: str,
) -> dict[str, Any]:
    """Return `ds.make` kwargs for the reach task."""
    if robot_preset == "franka_with_gripper_normalized":
        return {
            "sim": ds.options.SimOptions(control_hz=20, n_envs=n_envs),
            "robot": ds.options.RobotOptions(
                single=ds.options.ArmOptions(
                    manipulator="franka_with_gripper",
                    manipulator_controller=ds.options.ControllerOptions(
                        name="joint_position",
                        config={"normalized": True},
                    ),
                ),
            ),
            "render_mode": render_mode,
            "cameras": None,
        }
    if robot_preset == "legacy_franka_robotiq":
        return {
            "manipulator": "franka",
            "gripper": "robotiq",
            "arm_control": "joint_position",
            "gripper_control": "joint_position",
            "n_envs": n_envs,
            "render_mode": render_mode,
            "cameras": None,
        }
    raise ValueError(f"Unknown reach robot preset: {robot_preset}")


def _lift_make_kwargs(
    *,
    n_envs: int,
    render_mode: str | None,
    robot_preset: str,
) -> dict[str, Any]:
    """Return `ds.make` kwargs for the lift task."""
    if robot_preset in {
        "franka_umi_tactile_joint_position",
        "franka_umi_tactile_abs",
    }:
        return {
            "sim": ds.options.SimOptions(control_hz=20, n_envs=n_envs),
            "robot": ds.options.RobotOptions(
                type_of_robot="single",
                visualize_tcp=False,
                single=ds.options.ArmOptions(
                    manipulator="franka",
                    gripper="umi",
                    manipulator_controller=ds.options.ControllerOptions(
                        name="joint_position",
                        config={"normalized": True, "clip": True},
                    ),
                    gripper_controller=ds.options.ControllerOptions(
                        name="joint_position",
                        config={"normalized": True, "clip": True},
                    ),
                ),
            ),
            "render_mode": render_mode,
            "cameras": None,
        }
    raise ValueError(f"Unknown lift robot preset: {robot_preset}")


def make_dexsuite_env(
    env_name: str,
    *,
    n_envs: int,
    render_mode: str | None = None,
    robot_preset: str | None = None,
):
    """Build a raw DexSuite env from centralized presets."""
    env_cfg = ENV_CONFIGS[env_name]
    preset = robot_preset or env_cfg.default_robot_preset
    if env_name == "reach":
        return ds.make(
            env_cfg.task,
            **_reach_make_kwargs(
                n_envs=n_envs,
                render_mode=render_mode,
                robot_preset=preset,
            ),
        )
    if env_name == "lift":
        env = ds.make(
            env_cfg.task,
            **_lift_make_kwargs(
                n_envs=n_envs,
                render_mode=render_mode,
                robot_preset=preset,
            ),
        )
        if hasattr(env.robot, "tactile") and env.robot.tactile is not None:
            env.robot.tactile_summary_only = True
        return env
    raise ValueError(f"No SKRL environment builder registered for: {env_name}")


def make_skrl_env(
    env_name: str,
    *,
    n_envs: int,
    render_mode: str | None = None,
    robot_preset: str | None = None,
):
    """Build and wrap a DexSuite environment for SKRL."""
    env_cfg = ENV_CONFIGS[env_name]
    env = make_dexsuite_env(
        env_name,
        n_envs=n_envs,
        render_mode=render_mode,
        robot_preset=robot_preset,
    )
    if getattr(env, "n_envs", 1) != int(n_envs):
        raise RuntimeError(
            f"Requested n_envs={n_envs} but env.n_envs={getattr(env, 'n_envs', None)}.",
        )
    return WRAPPERS[env_cfg.wrapper](env)
