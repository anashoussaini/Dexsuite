"""Default builder spec generation.

This module centralizes how the interactive builder chooses its initial state.
"""

from __future__ import annotations

if __package__:
    from scripts.interactive_builder.spec import ArmSpec, BuilderSpec
else:
    from spec import ArmSpec, BuilderSpec


def default_spec() -> BuilderSpec:
    """Return a reasonable default BuilderSpec.

    Preference order:
    1) Use the dexsuite.options dataclasses (ensures consistency with YAML).
    2) Fall back to common hard-coded defaults if options cannot be imported.
    """
    try:
        from dexsuite.options import RobotOptions, SimOptions
    except ImportError:
        return BuilderSpec(
            task="reach",
            type_of_robot="single",
            single=ArmSpec(
                manipulator="franka",
                gripper="robotiq",
                arm_control="osc_pose",
                gripper_control="joint_position",
            ),
            control_hz=20,
            n_envs=1,
            cameras=None,
            render_mode="human",
            input_device="keyboard",
        )

    ro = RobotOptions()
    sim = SimOptions()

    if ro.type_of_robot == "single":
        arm = ro.single
        if arm is None:
            raise ValueError("Default RobotOptions.single is required for single mode.")
        return BuilderSpec(
            task="reach",
            type_of_robot="single",
            single=ArmSpec(
                manipulator=arm.manipulator,
                gripper=arm.gripper,
                arm_control=arm.manipulator_controller.name,
                gripper_control=(
                    None
                    if arm.gripper_controller is None
                    else arm.gripper_controller.name
                ),
                arm_control_config=dict(arm.manipulator_controller.config),
                gripper_control_config=(
                    {}
                    if arm.gripper_controller is None
                    else dict(arm.gripper_controller.config)
                ),
            ),
            control_hz=sim.control_hz,
            n_envs=sim.n_envs,
            performance_mode=sim.performance_mode,
            cameras=None,
            render_mode="human",
            input_device="keyboard",
        )

    left = ro.left
    right = ro.right
    if left is None or right is None:
        raise ValueError(
            "Default RobotOptions.left/right are required for bimanual mode.",
        )
    return BuilderSpec(
        task="bimanual_reach",
        type_of_robot="bimanual",
        left=ArmSpec(
            manipulator=left.manipulator,
            gripper=left.gripper,
            arm_control=left.manipulator_controller.name,
            gripper_control=(
                None
                if left.gripper_controller is None
                else left.gripper_controller.name
            ),
            arm_control_config=dict(left.manipulator_controller.config),
            gripper_control_config=(
                {}
                if left.gripper_controller is None
                else dict(left.gripper_controller.config)
            ),
        ),
        right=ArmSpec(
            manipulator=right.manipulator,
            gripper=right.gripper,
            arm_control=right.manipulator_controller.name,
            gripper_control=(
                None
                if right.gripper_controller is None
                else right.gripper_controller.name
            ),
            arm_control_config=dict(right.manipulator_controller.config),
            gripper_control_config=(
                {}
                if right.gripper_controller is None
                else dict(right.gripper_controller.config)
            ),
        ),
        layout_preset=ro.layout.preset,
        control_hz=sim.control_hz,
        n_envs=sim.n_envs,
        performance_mode=sim.performance_mode,
        cameras=None,
        render_mode="human",
        input_device="none",
    )
