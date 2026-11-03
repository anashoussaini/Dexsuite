"""Configuration options for robot hardware, controllers, and layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from dexsuite.utils.options_utils import load_config, resolve_workspace_strict

from .layout import LayoutOptions, PoseOptions
from .options import Vec3


# --------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_controller_defaults() -> tuple[str, str, dict[str, dict[str, Any]]]:
    """Load default controller names and configurations from YAML files.

    Returns:
        A tuple containing the default manipulator controller name, the default
        gripper controller name, and a dictionary of all available controller
        specifications.

    Raises:
        TypeError: If a controllers.yaml section has the wrong type.
        ValueError: If the controllers.yaml file is missing required sections.
    """
    cfg = load_config("controllers")
    if not isinstance(cfg, dict):
        raise TypeError("controllers.yaml must be a mapping.")
    if "defaults" not in cfg or "controllers" not in cfg:
        raise ValueError(
            "controllers.yaml must contain 'defaults' and 'controllers' sections.",
        )
    dflt = cfg["defaults"]
    ctrls = cfg["controllers"]
    if "manipulator" not in dflt or "gripper" not in dflt:
        raise ValueError(
            "controllers.yaml.defaults must have 'manipulator' and 'gripper'.",
        )
    if not isinstance(ctrls, dict):
        raise TypeError(
            "controllers.yaml.controllers must be a mapping of controller_name "
            "-> params.",
        )
    return str(dflt["manipulator"]), str(dflt["gripper"]), ctrls


_DEFAULT_MANIPULATOR = "franka"
_DEFAULT_GRIPPER = "robotiq"
_DEFAULT_ROBOT_TYPE = "single"
_DEFAULT_VISUALIZE_TCP = False
_DEFAULT_VISUALIZE_AABB = False


def _default_manipulator_controller() -> ControllerOptions:
    manip_ctrl, _grip_ctrl, specs = _load_controller_defaults()
    return ControllerOptions(
        name=manip_ctrl,
        config=dict(specs.get(manip_ctrl, {})),
    )


def _default_gripper_controller() -> ControllerOptions:
    _manip_ctrl, grip_ctrl, specs = _load_controller_defaults()
    return ControllerOptions(
        name=grip_ctrl,
        config=dict(specs.get(grip_ctrl, {})),
    )


@dataclass(slots=True)
class AABBOptions:
    """An axis-aligned bounding box (AABB) in the arm's base frame.

    Attributes:
        min: The (x, y, z) coordinates of the minimum corner of the box.
        max: The (x, y, z) coordinates of the maximum corner of the box.
    """

    min: Vec3 = (0.0, 0.0, 0.0)
    max: Vec3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        """Validate that the bounding box is numeric and ordered."""
        if len(self.min) != 3 or len(self.max) != 3:
            raise ValueError("AABB must have 3D min and max.")
        for v in (*self.min, *self.max):
            if not isinstance(v, (int, float)):
                raise TypeError("AABB coordinates must be numeric.")
        if any(lo > hi for lo, hi in zip(self.min, self.max, strict=False)):
            raise ValueError("AABBOptions: each min[i] must be <= max[i].")


@dataclass(slots=True)
class ControllerOptions:
    """Specifications for a controller and its configuration.

    Attributes:
        name: The name of the controller, which must be registered in the
            controller registry.
        config: A dictionary of parameters to be passed to the controller's
            __init__ method during instantiation.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArmOptions:
    """Specifications for a single robotic arm, including its gripper and controllers.

    This class specifies the hardware models for a manipulator and an optional
    gripper, as well as the controllers used to operate them. For modular
    arms, a gripper model must be specified. For integrated arms, the gripper
    field should be None or "builtin".

    Attributes:
        manipulator: The registered name of the manipulator model.
        gripper: The registered name of the gripper model, if applicable.
        manipulator_controller: The controller configuration for the arm.
        gripper_controller: The controller configuration for the gripper.
        workspace: An optional AABB defining the robot's operational workspace.
    """

    manipulator: str = _DEFAULT_MANIPULATOR
    gripper: str | None = _DEFAULT_GRIPPER

    manipulator_controller: ControllerOptions = field(
        default_factory=_default_manipulator_controller,
    )
    gripper_controller: ControllerOptions | None = field(
        default_factory=_default_gripper_controller,
    )

    workspace: AABBOptions | None = None

    def __post_init__(self) -> None:
        """Resolve a default workspace for known manipulators."""
        if self.workspace is None:
            try:
                spec = resolve_workspace_strict(self.manipulator)
            except ValueError:
                # Allow constructing options for unknown manipulators so errors can
                # be raised at build time (registry lookup) rather than at options
                # construction time.
                return
            self.workspace = AABBOptions(min=tuple(spec["min"]), max=tuple(spec["max"]))


@dataclass(slots=True)
class RobotOptions:
    """The complete robot configuration for an environment.

    This class supports both single-arm and bimanual setups. For bimanual
    robots, left and right arm options must be provided, along with a layout
    preset. For single-arm robots, only the single arm option is used.

    Attributes:
        type_of_robot: The robot configuration type, either 'single' or
            'bimanual'.
        single: The configuration for a single-arm robot.
        left: The configuration for the left arm of a bimanual robot.
        right: The configuration for the right arm of a bimanual robot.
        layout: The placement options for the robot(s) in the world.
        visualize_tcp: If True, renders a visual marker for the Tool Center
            Point (TCP).
        visualize_aabb: If True, renders a visual marker for the workspace
            bounding box (AABB).
    """

    type_of_robot: Literal["single", "bimanual"] = _DEFAULT_ROBOT_TYPE
    single: ArmOptions | None = None
    left: ArmOptions | None = None
    right: ArmOptions | None = None

    layout: LayoutOptions = field(default_factory=LayoutOptions)

    visualize_tcp: bool = _DEFAULT_VISUALIZE_TCP
    visualize_aabb: bool = _DEFAULT_VISUALIZE_AABB

    def __post_init__(self) -> None:
        """Normalize and validate single-arm or bimanual robot options."""
        if self.type_of_robot == "single":
            self._post_init_single()
            return

        if self.type_of_robot == "bimanual":
            self._post_init_bimanual()
            return

        raise ValueError(f"Unknown type_of_robot '{self.type_of_robot}'.")

    def _post_init_single(self) -> None:
        if self.single is None:
            self.single = ArmOptions()
        if self.layout.single is None:
            self.layout.single = PoseOptions()
        if self.layout.preset is not None:
            raise ValueError(
                "Single-arm robots do not use layout presets "
                "(set 'single' pose explicitly if needed).",
            )

    def _post_init_bimanual(self) -> None:
        if self.left is None:
            self.left = ArmOptions()
        if self.right is None:
            self.right = ArmOptions()

        user_left = self.layout.left is not None
        user_right = self.layout.right is not None
        if user_left and user_right:
            return

        if not self.layout.preset:
            self.layout.preset = "side_by_side"

        left_preset, right_preset = self.layout.resolve_bimanual(
            self.left.manipulator,
            self.right.manipulator,
        )

        if not user_left:
            self.layout.left = left_preset
        if not user_right:
            self.layout.right = right_preset
