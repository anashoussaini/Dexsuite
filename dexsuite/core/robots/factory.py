"""Create robot instances from configuration options."""

from __future__ import annotations

import torch

from dexsuite.core.components.arm import (
    IntegratedManipulatorModel,
    ModularManipulatorModel,
)
from dexsuite.core.registry import GRIP_REG, MANIP_REG
from dexsuite.core.robots.integrated_robot import IntegratedRobot
from dexsuite.core.robots.modular_robot import ModularRobot
from dexsuite.options import ArmOptions, ControllerOptions, PoseOptions, RobotOptions
from dexsuite.utils import rpy_to_quat

__all__ = ["make_arm_from_options", "make_robot_from_options"]


def _ensure_normalized(opt: ControllerOptions | None) -> ControllerOptions | None:
    """Ensure that controller actions are normalized by default.

    This helper function sets normalized=True in the controller configuration
    unless it has been explicitly set by the user. This avoids the need for
    hardware limits when defining the action space, especially for torque control.

    Args:
        opt: The controller options to modify.

    Returns:
        The modified controller options, or None if the input was None.
    """
    if opt is None:
        return None
    config = dict(opt.config or {})
    config.setdefault("normalized", True)
    return ControllerOptions(name=opt.name, config=config)


def make_robot_from_options(*, robot_options: RobotOptions, scene):
    """Create a single-arm or bimanual robot from high-level options.

    This factory function reads the robot_options and instantiates the
    appropriate robot class (IntegratedRobot, ModularRobot, or BimanualRobot)
    with the specified configurations.

    Args:
        robot_options: The main robot configuration object.
        scene: The simulation scene to add the robot to.

    Returns:
        An instance of a robot class corresponding to the provided options.

    Raises:
        ValueError: If a required option (e.g., manipulator) is not specified.
        TypeError: If an unknown type_of_robot is provided.
    """
    if robot_options.type_of_robot == "single":
        pose_options = robot_options.layout.single
        arm_options = robot_options.single
        if arm_options.manipulator is None:
            raise ValueError("A manipulator has to be defined")

        return make_arm_from_options(
            arm_options=arm_options,
            scene=scene,
            pose_options=pose_options,
            visualize_tcp=robot_options.visualize_tcp,
        )

    if robot_options.type_of_robot == "bimanual":
        pose_left = robot_options.layout.left
        pose_right = robot_options.layout.right
        arm_options_left = robot_options.left
        arm_options_right = robot_options.right

        from dexsuite.core.robots.bimanual_robot import BimanualRobot

        return BimanualRobot(
            arm_options=(arm_options_left, arm_options_right),
            scene=scene,
            pose_options=(pose_left, pose_right),
            visualize_tcp=robot_options.visualize_tcp,
        )

    raise TypeError(
        "type_of_robot should be 'single' or 'bimanual'. "
        f"(Got {robot_options.type_of_robot})",
    )


def make_arm_from_options(
    *,
    arm_options: ArmOptions,
    scene,
    pose_options: PoseOptions,
    visualize_tcp: bool = True,
):
    """Create a single-arm robot (modular or integrated) from options.

    This function instantiates a manipulator model from the registry and
    determines whether to create a ModularRobot or an IntegratedRobot
    based on the manipulator's type.

    Args:
        arm_options: The configuration options for the arm.
        scene: The simulation scene to add the robot to.
        pose_options: The placement options for the robot's base.
        visualize_tcp: If True, enables visualization of the Tool Center Point.

    Returns:
        An instance of ModularRobot or IntegratedRobot.

    Raises:
        ValueError: If a specified manipulator or gripper is not found in the
            registries, or if the controller configuration is invalid.
        TypeError: If the manipulator model is of an unknown subclass.
    """
    manip_cls = MANIP_REG.get(arm_options.manipulator)
    if manip_cls is None:
        raise ValueError(f"Unknown manipulator ({arm_options.manipulator})")
    manipulator = manip_cls()

    base_pos = pose_options.pos
    if pose_options.quat is not None:
        quat = pose_options.quat
    elif pose_options.yaw_rad is not None:
        quat = rpy_to_quat(torch.tensor([0.0, 0.0, float(pose_options.yaw_rad)])).cpu()
    else:
        raise ValueError("rotation is not defined")

    if isinstance(manipulator, IntegratedManipulatorModel):
        return IntegratedRobot(
            manipulator=manipulator,
            arm_control_options=_ensure_normalized(arm_options.manipulator_controller),
            gripper_control_options=_ensure_normalized(arm_options.gripper_controller),
            scene=scene,
            base_pos=base_pos,
            quat=quat,
            visualize_tcp=visualize_tcp,
        )

    if isinstance(manipulator, ModularManipulatorModel):
        grip_name = arm_options.gripper
        if grip_name is None:
            available_adapters = ", ".join(sorted(manipulator.adapters)) or "<none>"
            raise ValueError(
                f"Modular manipulator '{type(manipulator).__name__}' requires a gripper. "
                f"Available adapters: {available_adapters}.",
            )

        grip_cls = GRIP_REG.get(grip_name)
        if grip_cls is None:
            available_grippers = ", ".join(sorted(GRIP_REG)) or "<none>"
            raise ValueError(
                f"Modular manipulator '{type(manipulator).__name__}' requires a known gripper model "
                f"(got {grip_name}). Available grippers: {available_grippers}.",
            )
        gripper = grip_cls()
        gripper._registry_name = grip_name

        _validate_gripper_controller(arm_options.gripper_controller)

        return ModularRobot(
            manipulator=manipulator,
            gripper=gripper,
            arm_control_options=_ensure_normalized(arm_options.manipulator_controller),
            gripper_control_options=_ensure_normalized(arm_options.gripper_controller),
            scene=scene,
            base_pos=base_pos,
            quat=quat,
            visualize_tcp=visualize_tcp,
        )

    raise TypeError(f"Unknown manipulator subclass ({type(manipulator).__name__})")


def _validate_gripper_controller(
    gripper_controller: ControllerOptions | None,
) -> None:
    if gripper_controller is not None and gripper_controller.name not in {
        "joint_position",
        "joint_velocity",
        "joint_torque",
    }:
        raise ValueError(
            f"Invalid gripper controller '{gripper_controller.name}'. Grippers only support "
            "joint_position | joint_velocity | joint_torque.",
        )
