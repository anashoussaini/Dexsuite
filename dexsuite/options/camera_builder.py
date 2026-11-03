"""Build camera option objects from direct camera arguments."""

from __future__ import annotations

from typing import Any

from .camera import CamerasOptions


def build_camera_options(
    kw: dict[str, Any],
    existing: CamerasOptions | None,
) -> CamerasOptions:
    """Return camera options resolved from object or direct arguments.

    Args:
        kw: Normalized `ds.make` keyword arguments.
        existing: Explicit camera options object, or `None`.

    Returns:
        Camera options used to construct the environment.
    """
    if "cameras" in kw and kw["cameras"] is None:
        return CamerasOptions(
            static={},
            dynamic={},
            modalities=normalize_modalities(kw.get("modalities", ("rgb",))),
        )

    if isinstance(existing, CamerasOptions):
        return existing

    names = kw.get("cameras")
    if names is None:
        names = ("front", "wrist")
    if not isinstance(names, (list, tuple)):
        raise TypeError(
            "cameras must be a list/tuple of names, e.g., ['front','wrist'].",
        )

    modalities = normalize_modalities(kw.get("modalities", ("rgb",)))
    static_map, dynamic_map = camera_maps_from_names(names)
    return CamerasOptions(static=static_map, dynamic=dynamic_map, modalities=modalities)


def normalize_modalities(modalities: Any) -> tuple[str, ...]:
    """Return normalized camera modality names.

    Args:
        modalities: Camera modality name or sequence of names.

    Returns:
        Tuple of modality names.

    Raises:
        TypeError: When modalities is not a string or sequence.
        ValueError: When RGB is not requested.
    """
    if isinstance(modalities, str):
        modalities = (modalities,)
    if not isinstance(modalities, (list, tuple)):
        raise TypeError("modalities must be a list/tuple (or a single string).")
    normalized = tuple(modalities)
    if "rgb" not in normalized:
        raise ValueError("modalities must include 'rgb'.")
    return normalized


def camera_maps_from_names(
    names: tuple[str, ...] | list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split public camera names into static and dynamic camera maps.

    Args:
        names: Camera names requested through the direct API.

    Returns:
        Static and dynamic camera preset maps keyed by public camera name.
    """
    static_map: dict[str, str] = {}
    dynamic_map: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"Camera name must be str, got {type(name)}")
        if name == "wrist":
            dynamic_map["wrist"] = "wrist_cam"
        else:
            static_map[name] = name

    return static_map, dynamic_map


__all__ = ["build_camera_options", "camera_maps_from_names", "normalize_modalities"]
