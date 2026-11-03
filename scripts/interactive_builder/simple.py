"""Non-curses fallback builder (plain stdin prompts).

This exists primarily for platforms where curses is unavailable.
"""

from __future__ import annotations

if __package__:
    from scripts.interactive_builder.defaults import default_spec
    from scripts.interactive_builder.registry_scan import (
        available_controllers,
        available_grippers,
        available_manipulators,
        available_tasks,
        manipulator_infos,
        supported_grippers_for_manipulator,
    )
    from scripts.interactive_builder.spec import (
        POSE_BACKENDS,
        ArmSpec,
        BuilderSpec,
        arm_backend,
        controller_supports_backend,
        set_arm_backend,
    )
else:
    from defaults import default_spec
    from registry_scan import (
        available_controllers,
        available_grippers,
        available_manipulators,
        available_tasks,
        manipulator_infos,
        supported_grippers_for_manipulator,
    )
    from spec import (
        POSE_BACKENDS,
        ArmSpec,
        BuilderSpec,
        arm_backend,
        controller_supports_backend,
        set_arm_backend,
    )


def _choose(prompt: str, options: list[str], *, default: str | None = None) -> str:
    if not options:
        raise ValueError(f"No options available for {prompt!r}")
    idx_default = options.index(default) if (default in options) else 0
    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(options, start=1):
            mark = "*" if (i - 1) == idx_default else " "
            print(f"  {mark} {i:2d}) {opt}")
        raw = input(f"Select [1-{len(options)}] (default {idx_default + 1}): ").strip()
        if raw == "":
            return options[idx_default]
        try:
            k = int(raw)
        except ValueError:
            continue
        if 1 <= k <= len(options):
            return options[k - 1]


def _default_arm(manipulators: list[str], grippers: list[str]) -> ArmSpec:
    return ArmSpec(
        manipulator=manipulators[0] if manipulators else "franka",
        gripper=grippers[0] if grippers else None,
        arm_control="osc_pose",
        gripper_control="joint_position",
    )


def _edit_arm(
    label: str,
    arm: ArmSpec | None,
    *,
    manipulators: list[str],
    grippers: list[str],
    controllers: list[str],
    manip_kinds: dict[str, str],
) -> ArmSpec:
    if arm is None:
        arm = _default_arm(manipulators, grippers)

    arm.manipulator = _choose(
        f"{label} manipulator",
        manipulators,
        default=arm.manipulator,
    )
    if manip_kinds.get(arm.manipulator) == "integrated":
        arm.gripper = None
    else:
        supported = supported_grippers_for_manipulator(arm.manipulator)
        grip_opts = supported if supported else grippers
        if grip_opts:
            arm.gripper = _choose(
                f"{label} gripper",
                grip_opts,
                default=arm.gripper,
            )

    prev_arm_control = arm.arm_control
    arm.arm_control = _choose(
        f"{label} arm controller",
        controllers,
        default=arm.arm_control,
    )
    if arm.arm_control != prev_arm_control:
        arm.arm_control_config.clear()

    _edit_arm_backend(label, arm)
    _edit_gripper_controller(label, arm, controllers)
    return arm


def _edit_arm_backend(label: str, arm: ArmSpec) -> None:
    if controller_supports_backend(arm.arm_control):
        set_arm_backend(
            arm,
            _choose(
                f"{label} arm controller backend",
                list(POSE_BACKENDS),
                default=arm_backend(arm),
            ),
        )
        return
    set_arm_backend(arm, POSE_BACKENDS[0])


def _edit_gripper_controller(
    label: str,
    arm: ArmSpec,
    controllers: list[str],
) -> None:
    grip_ctrls = [controller for controller in controllers if controller.startswith("joint_")]
    if not grip_ctrls:
        return
    arm.gripper_control = _choose(
        f"{label} gripper controller",
        grip_ctrls,
        default=arm.gripper_control,
    )


def _edit_robot(spec: BuilderSpec) -> None:
    spec.type_of_robot = _choose(
        "Robot type",
        ["single", "bimanual"],
        default=spec.type_of_robot,
    )  # type: ignore[assignment]

    manip_kinds = {info.key: info.kind for info in manipulator_infos()}
    manipulators = available_manipulators()
    grippers = available_grippers()
    controllers = available_controllers()
    edit_args = {
        "manipulators": manipulators,
        "grippers": grippers,
        "controllers": controllers,
        "manip_kinds": manip_kinds,
    }

    if spec.type_of_robot == "single":
        spec.single = _edit_arm("Single", spec.single, **edit_args)
        spec.left = None
        spec.right = None
        return

    spec.left = _edit_arm("Left", spec.left, **edit_args)
    spec.right = _edit_arm("Right", spec.right, **edit_args)
    spec.single = None


def _edit_sim(spec: BuilderSpec) -> None:
    raw = input(f"\nControl Hz (current {spec.control_hz}): ").strip()
    if raw:
        spec.control_hz = int(raw)
    raw = input(f"n_envs (current {spec.n_envs}): ").strip()
    if raw:
        spec.n_envs = int(raw)
    raw = input(
        f"performance_mode (0/1, current {int(spec.performance_mode)}): ",
    ).strip()
    if raw:
        spec.performance_mode = bool(int(raw))


def _edit_input_device(spec: BuilderSpec) -> None:
    raw = input(
        f"input_device [keyboard|spacemouse|vive_controller|vive_tracker|none] (current {spec.input_device}): ",
    ).strip()
    if raw:
        spec.input_device = raw


def run_simple_builder() -> BuilderSpec:
    """Run a basic prompt-based builder and return a BuilderSpec."""
    spec = default_spec()

    tasks = available_tasks()
    if tasks:
        spec.task = _choose(
            "Task",
            tasks,
            default=spec.task if spec.task in tasks else None,
        )

    _edit_robot(spec)
    _edit_sim(spec)
    _edit_input_device(spec)
    return spec
