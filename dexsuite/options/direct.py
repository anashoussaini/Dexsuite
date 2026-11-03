"""Build robot and simulation options from direct `ds.make` arguments."""

from __future__ import annotations

from typing import Any

from dexsuite.utils.options_utils import load_config

from .layout import LayoutOptions
from .robot import ArmOptions, ControllerOptions, RobotOptions
from .sim import SimOptions


def build_robot_options(
    kw: dict[str, Any],
    existing: RobotOptions | None,
) -> RobotOptions:
    """Return robot options resolved from object or direct arguments.

    Args:
        kw: Normalized `ds.make` keyword arguments.
        existing: Explicit robot options object, or `None`.

    Returns:
        Robot options used to construct the environment.

    Raises:
        ValueError: When required direct robot arguments are missing or malformed.
    """
    if isinstance(existing, RobotOptions):
        return existing

    manipulator = kw.get("manipulator")
    if manipulator is None:
        raise ValueError("Missing required 'manipulator' argument.")

    manipulator_tuple = _as_tuple(manipulator)
    if len(manipulator_tuple) == 1:
        return _build_single_robot_options(kw, manipulator_tuple[0])
    if len(manipulator_tuple) == 2:
        return _build_bimanual_robot_options(kw, manipulator_tuple)

    raise ValueError(
        "manipulator must be a string (single) or a length-2 pair (bimanual).",
    )


def build_sim_options(kw: dict[str, Any], existing: SimOptions | None) -> SimOptions:
    """Return simulation options resolved from object or direct arguments.

    Args:
        kw: Normalized `ds.make` keyword arguments.
        existing: Explicit simulation options object, or `None`.

    Returns:
        Simulation options used to construct the environment.
    """
    if isinstance(existing, SimOptions):
        return existing

    defaults = SimOptions()
    return SimOptions(
        control_hz=int(kw.get("control_hz", defaults.control_hz)),
        performance_mode=bool(kw.get("performance_mode", defaults.performance_mode)),
        n_envs=int(kw.get("n_envs", defaults.n_envs)),
    )


def load_controller_specs() -> dict[str, dict[str, Any]]:
    """Return controller specifications from the package configuration.

    Returns:
        Mapping from controller name to default controller config.

    Raises:
        TypeError: When the controller config file is not a mapping.
    """
    cfg = load_config("controllers")
    if not isinstance(cfg, dict):
        raise TypeError("controllers.yaml must be a mapping.")
    controllers = cfg.get("controllers")
    if not isinstance(controllers, dict):
        raise TypeError("controllers.yaml must contain a 'controllers' mapping.")
    return controllers


def controller_options_from_name(name: str | None) -> ControllerOptions | None:
    """Return controller options loaded from config defaults.

    Args:
        name: Controller name, or `None`.

    Returns:
        Controller options for the requested controller, or `None`.
    """
    if name is None:
        return None
    specs = load_controller_specs()
    params = dict(specs.get(name, {}))
    return ControllerOptions(name=name, config=params)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _build_single_robot_options(kw: dict[str, Any], manipulator: Any) -> RobotOptions:
    gripper = kw.get("gripper")
    if isinstance(gripper, (tuple, list)):
        gripper = gripper[0]

    defaults = ArmOptions()
    single = ArmOptions(
        manipulator=str(manipulator),
        gripper=gripper,
        manipulator_controller=controller_options_from_name(kw.get("arm_control"))
        or defaults.manipulator_controller,
        gripper_controller=controller_options_from_name(kw.get("gripper_control"))
        or defaults.gripper_controller,
    )
    return RobotOptions(type_of_robot="single", single=single)


def _build_bimanual_robot_options(
    kw: dict[str, Any],
    manipulator_pair: tuple[Any, ...],
) -> RobotOptions:
    gripper_pair = _resolve_bimanual_gripper_pair(kw.get("gripper"))
    defaults = ArmOptions()
    arm_controller = (
        controller_options_from_name(kw.get("arm_control"))
        or defaults.manipulator_controller
    )
    gripper_controller = (
        controller_options_from_name(kw.get("gripper_control"))
        or defaults.gripper_controller
    )

    left = ArmOptions(
        manipulator=str(manipulator_pair[0]),
        gripper=gripper_pair[0],
        manipulator_controller=arm_controller,
        gripper_controller=gripper_controller,
    )
    right = ArmOptions(
        manipulator=str(manipulator_pair[1]),
        gripper=gripper_pair[1],
        manipulator_controller=arm_controller,
        gripper_controller=gripper_controller,
    )
    layout_preset = kw.get("layout")
    layout = (
        LayoutOptions(preset=layout_preset)
        if isinstance(layout_preset, str)
        else LayoutOptions()
    )
    return RobotOptions(type_of_robot="bimanual", left=left, right=right, layout=layout)


def _resolve_bimanual_gripper_pair(gripper: Any) -> tuple[Any, Any]:
    if isinstance(gripper, (tuple, list)):
        if len(gripper) != 2:
            raise ValueError(
                "For bimanual, 'gripper' must be a string (replicated) "
                "or a length-2 pair.",
            )
        return str(gripper[0]), str(gripper[1])
    return gripper, gripper


__all__ = [
    "build_robot_options",
    "build_sim_options",
    "controller_options_from_name",
    "load_controller_specs",
]
