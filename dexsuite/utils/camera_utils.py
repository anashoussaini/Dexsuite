"""Normalize camera presets, target names, and transforms."""

from __future__ import annotations

from typing import Any

import torch


def normalize_camera_target_id(obj: object, fallback: str) -> str:
    """Return a normalized identifier for matching camera presets.

    Args:
        obj: Object or class to identify.
        fallback: Fallback string when no name is available.

    Returns:
        Lowercase identifier with surrounding whitespace removed.
    """
    registry_name = getattr(obj, "_registry_name", None)
    name = registry_name or getattr(obj, "__name__", None)
    if name is None:
        name = obj.__class__.__name__ or fallback
    return str(name).strip().lower()


def parse_bimanual_camera_side(cam_name: str) -> str | None:
    """Infer a bimanual side marker from a camera name.

    Args:
        cam_name: Camera name requested by the user.

    Returns:
        `left` or `right` when present, otherwise `None`.
    """
    normalized_name = str(cam_name).strip().lower()
    if normalized_name.startswith("left_") or normalized_name.endswith("_left"):
        return "left"
    if normalized_name.startswith("right_") or normalized_name.endswith("_right"):
        return "right"
    return None


def dynamic_camera_targets(
    requested_name: str,
    *,
    is_bimanual: bool,
    single_label: str,
) -> list[tuple[str, str]]:
    """Return concrete camera names and gripper labels for a dynamic request.

    Args:
        requested_name: Public camera name from the camera options.
        is_bimanual: Whether both left and right gripper labels are present.
        single_label: Gripper label used for single-arm robots.

    Returns:
        Pairs of concrete camera name and target gripper label.
    """
    if not is_bimanual:
        return [(requested_name, single_label)]

    side = parse_bimanual_camera_side(requested_name)
    if side in ("left", "right"):
        return [(requested_name, side)]
    return [(f"left_{requested_name}", "left"), (f"right_{requested_name}", "right")]


def dynamic_camera_overrides(preset: dict[str, Any], entry: str) -> dict[str, object]:
    """Return normalized dynamic-camera override blocks from a preset.

    Args:
        preset: Raw dynamic camera preset mapping.
        entry: Preset name used in validation errors.

    Returns:
        Override mapping keyed by normalized manipulator or gripper id.

    Raises:
        TypeError: When the override block is not a mapping.
    """
    raw_overrides = preset.get("overrides")
    if raw_overrides is None:
        raw_overrides = preset.get("by")
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        raise TypeError(
            f"Dynamic preset '{entry}' overrides must be a mapping.",
        )
    return {str(key).strip().lower(): value for key, value in raw_overrides.items()}


def select_dynamic_camera_override(
    *,
    entry: str,
    default: dict[str, Any],
    overrides: dict[str, object],
    manip_obj: object,
    gripper_obj: object,
) -> dict[str, Any]:
    """Select the camera override matching the current robot components.

    Args:
        entry: Preset name used in validation errors.
        default: Default preset block.
        overrides: Normalized override blocks from the preset.
        manip_obj: Manipulator object used for integrated-gripper matching.
        gripper_obj: Gripper object used for modular-gripper matching.

    Returns:
        Selected override mapping.

    Raises:
        TypeError: When the selected override is not a mapping.
        ValueError: When no usable override or default block is available.
    """
    manip_id = normalize_camera_target_id(manip_obj, "manip")
    choice = None
    if hasattr(gripper_obj, "_registry_name"):
        gripper_id = normalize_camera_target_id(gripper_obj, "gripper")
        choice = overrides.get(gripper_id)
        if choice is None:
            canonical_id = normalize_camera_target_id(gripper_obj.__class__, gripper_id)
            if canonical_id != gripper_id:
                choice = overrides.get(canonical_id)
    if choice is None:
        choice = overrides.get(manip_id, default)
    if not isinstance(choice, dict):
        raise TypeError(
            f"Dynamic preset '{entry}' selected override must be a mapping.",
        )
    if not choice:
        raise ValueError(
            "Dynamic preset must provide either a matching override or a "
            "'default' block.",
        )
    return choice


def normalize_quat_wxyz(quat: object) -> tuple[float, float, float, float]:
    """Return a finite normalized WXYZ quaternion.

    Args:
        quat: Quaternion-like value with four entries.

    Returns:
        Normalized WXYZ quaternion, or identity when the input norm is invalid.
    """
    quat_tensor = torch.as_tensor(quat, dtype=torch.float64).reshape(4)
    quat_norm = torch.linalg.vector_norm(quat_tensor)
    if not torch.isfinite(quat_norm).item() or float(quat_norm) <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value) for value in (quat_tensor / quat_norm).tolist())


__all__ = [
    "dynamic_camera_overrides",
    "dynamic_camera_targets",
    "normalize_camera_target_id",
    "normalize_quat_wxyz",
    "parse_bimanual_camera_side",
    "select_dynamic_camera_override",
]
