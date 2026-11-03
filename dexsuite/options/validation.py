"""Validate and normalize public `ds.make` keyword arguments."""

from __future__ import annotations

from typing import Any

from .camera import CamerasOptions
from .layout import LayoutOptions
from .robot import RobotOptions
from .sim import SimOptions

_MAKE_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "robot",
        "sim",
        "cameras",
        "camera_gui",
        "gui",
        "layout",
        "render_mode",
        "manipulator",
        "gripper",
        "arm_control",
        "gripper_control",
        "control_hz",
        "performance_mode",
        "n_envs",
        "modalities",
        "n_env",
    },
)


def normalize_make_options(kw_in: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize `ds.make` keyword arguments.

    Args:
        kw_in: Keyword arguments passed to `ds.make`.

    Returns:
        Normalized keyword mapping with aliases resolved.
    """
    kw = dict(kw_in)

    if "n_env" in kw:
        if "n_envs" in kw:
            raise TypeError("ds.make() got both 'n_env' and 'n_envs'; use only one.")
        kw["n_envs"] = kw.pop("n_env")
    if "gui" in kw:
        if "camera_gui" in kw:
            raise TypeError(
                "ds.make() got both 'gui' and 'camera_gui'; use only one.",
            )
        kw["camera_gui"] = kw.pop("gui")

    unknown = set(kw) - set(_MAKE_ALLOWED_KEYS)
    if unknown:
        unknown_s = ", ".join(sorted(unknown))
        allowed_s = ", ".join(sorted(k for k in _MAKE_ALLOWED_KEYS if k != "n_env"))
        raise TypeError(
            f"ds.make() got unexpected keyword argument(s): {unknown_s}. "
            f"Allowed: {allowed_s}.",
        )

    if "cameras" in kw and isinstance(kw["cameras"], str):
        kw["cameras"] = (kw["cameras"],)
    if "modalities" in kw and isinstance(kw["modalities"], str):
        kw["modalities"] = (kw["modalities"],)

    _validate_make_inputs(kw)
    _validate_direct_robot_inputs(kw)
    return kw


def _validate_make_inputs(kw: dict[str, Any]) -> None:
    _validate_layout_input(kw)
    _validate_cameras_input(kw)
    _validate_camera_gui_input(kw)
    _validate_modalities_input(kw)
    _validate_robot_input_exclusivity(kw)
    _validate_sim_input_exclusivity(kw)
    _validate_scalar_inputs(kw)


def _validate_layout_input(kw: dict[str, Any]) -> None:
    layout = kw.get("layout")
    if layout is not None and not isinstance(layout, (LayoutOptions, str)):
        raise TypeError("layout must be a preset string, LayoutOptions, or None.")


def _validate_cameras_input(kw: dict[str, Any]) -> None:
    cameras = kw.get("cameras")
    if cameras is not None and not isinstance(cameras, (CamerasOptions, tuple, list)):
        raise TypeError(
            "cameras must be CamerasOptions, a list/tuple of names, "
            "a single name, or None.",
        )
    if isinstance(cameras, (tuple, list)):
        if len(cameras) == 0:
            raise ValueError("cameras must be a non-empty list/tuple (or None).")
        invalid_camera_names = [cam for cam in cameras if not isinstance(cam, str)]
        if invalid_camera_names:
            raise TypeError("All camera names must be strings.")

    if isinstance(cameras, CamerasOptions) and kw.get("modalities") is not None:
        raise ValueError(
            "When passing cameras=CamerasOptions(...), set modalities on that object "
            "and do not pass modalities=... separately.",
        )


def _validate_camera_gui_input(kw: dict[str, Any]) -> None:
    camera_gui = kw.get("camera_gui")
    if camera_gui is None or isinstance(camera_gui, bool):
        return
    if not isinstance(camera_gui, dict):
        raise TypeError(
            "camera_gui must be a bool, a dict of camera-name bools, or None.",
        )
    if any(not isinstance(name, str) for name in camera_gui):
        raise TypeError("camera_gui mapping keys must be strings.")
    if any(not isinstance(enabled, bool) for enabled in camera_gui.values()):
        raise TypeError("camera_gui mapping values must be bools.")


def _validate_modalities_input(kw: dict[str, Any]) -> None:
    modalities = kw.get("modalities")
    if modalities is not None and not isinstance(modalities, (tuple, list)):
        raise TypeError(
            "modalities must be a list/tuple of strings or a single string.",
        )
    if isinstance(modalities, (tuple, list)):
        invalid_modalities = [
            modality for modality in modalities if not isinstance(modality, str)
        ]
        if invalid_modalities:
            raise TypeError("All modalities entries must be strings.")


def _validate_robot_input_exclusivity(kw: dict[str, Any]) -> None:
    robot = kw.get("robot")
    if not isinstance(robot, RobotOptions):
        return

    direct_robot_keys = (
        "manipulator",
        "gripper",
        "arm_control",
        "gripper_control",
    )
    conflicting_robot_keys = [
        key for key in direct_robot_keys if kw.get(key) is not None
    ]
    if conflicting_robot_keys:
        raise ValueError(
            "Provide either 'robot=RobotOptions(...)' or the direct robot keys "
            f"{conflicting_robot_keys} - not both.",
        )
    if isinstance(kw.get("layout"), str):
        raise TypeError(
            "When passing robot=RobotOptions(...), do not pass layout as a string. "
            "Set robot.layout.preset or pass layout=LayoutOptions(...) instead.",
        )


def _validate_sim_input_exclusivity(kw: dict[str, Any]) -> None:
    sim = kw.get("sim")
    if not isinstance(sim, SimOptions):
        return

    direct_sim_keys = ("control_hz", "performance_mode", "n_envs")
    conflicting_sim_keys = [key for key in direct_sim_keys if kw.get(key) is not None]
    if conflicting_sim_keys:
        raise ValueError(
            "Provide either 'sim=SimOptions(...)' or the direct sim keys "
            f"{conflicting_sim_keys} - not both.",
        )


def _validate_scalar_inputs(kw: dict[str, Any]) -> None:
    _coerce_positive_int_field(kw, "control_hz")
    _coerce_positive_int_field(kw, "n_envs")
    if kw.get("performance_mode") is not None and not isinstance(
        kw["performance_mode"],
        bool,
    ):
        raise TypeError("performance_mode must be a bool.")

    render_mode = kw.get("render_mode")
    if render_mode is not None and not isinstance(render_mode, str):
        raise TypeError("render_mode must be a string or None.")
    if render_mode is not None and render_mode not in ("human", "rgb_array"):
        raise ValueError("render_mode must be 'human', 'rgb_array', or None.")


def _coerce_positive_int_field(kw: dict[str, Any], name: str) -> None:
    if name not in kw or kw[name] is None:
        return
    value = kw[name]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not a bool.")
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{name} must be an integer value, got {value!r}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an int, got {type(value)}.") from exc
    if int_value <= 0:
        raise ValueError(f"{name} must be > 0, got {int_value}.")
    kw[name] = int_value


def _validate_direct_robot_inputs(kw: dict[str, Any]) -> None:
    _validate_manipulator_input(kw.get("manipulator"))
    _validate_gripper_input(kw.get("gripper"))
    for name in ("arm_control", "gripper_control"):
        if kw.get(name) is not None and not isinstance(kw[name], str):
            raise TypeError(f"{name} must be a string or None.")


def _validate_manipulator_input(manipulator: Any) -> None:
    if manipulator is None:
        return
    if isinstance(manipulator, str):
        return
    if isinstance(manipulator, (tuple, list)):
        if len(manipulator) not in (1, 2):
            raise ValueError(
                "manipulator must be a string (single) or a length-2 pair "
                "(bimanual).",
            )
        invalid_manipulator_names = [
            name for name in manipulator if not isinstance(name, str)
        ]
        if invalid_manipulator_names:
            raise TypeError("manipulator entries must be strings.")
        return
    raise TypeError(
        "manipulator must be a string or a length-2 tuple/list of strings.",
    )


def _validate_gripper_input(gripper: Any) -> None:
    if gripper is None:
        return
    if isinstance(gripper, str):
        return
    if isinstance(gripper, (tuple, list)):
        if len(gripper) not in (1, 2):
            raise ValueError(
                "gripper must be a string/None (replicated) or a length-2 "
                "pair for bimanual.",
            )
        invalid_gripper_names = [
            name for name in gripper if name is not None and not isinstance(name, str)
        ]
        if invalid_gripper_names:
            raise TypeError("gripper entries must be strings or None.")
        return
    raise TypeError("gripper must be a string, None, or a tuple/list.")


__all__ = ["normalize_make_options"]
