"""Configuration options for static and dynamic cameras in the environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from dexsuite.utils.options_utils import load_config

from .options import Quat, ResHW, Vec3

_RENDER_MODALITIES: tuple[str, ...] = ("rgb", "depth", "segmentation", "normal")


@dataclass(slots=True)
class StaticCamOptions:
    """Define a fixed camera in world coordinates.

    Attributes:
        pos: The (x, y, z) position of the camera in the world frame.
        lookat: The (x, y, z) point in the world frame the camera looks at.
        fov: The vertical field of view in degrees. Must be between 0 and 180.
        res: The (width, height) resolution of the camera image in pixels.
        gui: Whether Genesis should create a GUI visualization for this camera.
        near: Rendering near clipping distance in metres (not sensor minimum range).
        far: Rendering far clipping distance in metres.
    """

    pos: Vec3
    lookat: Vec3
    fov: float = 55.0
    res: ResHW = (224, 224)
    gui: bool = False
    near: float = 0.1
    far: float = 20.0

    def __post_init__(self) -> None:
        """Validate the camera's field of view and resolution."""
        if not 0.0 < self.fov < 180.0:
            raise ValueError("StaticCamOptions.fov must be in (0, 180).")
        if min(self.res) <= 0:
            raise ValueError("StaticCamOptions.res must be positive integers.")
        if not 0.0 < self.near < self.far < float("inf"):
            raise ValueError("Camera clipping requires finite 0 < near < far.")
        if not isinstance(self.gui, bool):
            raise TypeError("StaticCamOptions.gui must be a bool.")


@dataclass(slots=True)
class DynamicCamOptions:
    """Define a camera that is attached to a robot link.

    Offsets are specified in the coordinate frame of the attached link. If link
    is not provided, the camera will be attached to the manipulator's end_link
    by default during environment setup.

    Attributes:
        link: The name of the robot link to attach the camera to.
        pos_offset: The (x, y, z) position offset relative to the link's frame.
        quat_offset: The (w, x, y, z) orientation offset (quaternion) relative
            to the link's frame.
        fov: The vertical field of view in degrees. Must be between 0 and 180.
        res: The (width, height) resolution of the camera image in pixels.
        gui: Whether Genesis should create a GUI visualization for this camera.
        near: Rendering near clipping distance in metres (not sensor minimum range).
        far: Rendering far clipping distance in metres.
    """

    link: str | None = None
    pos_offset: Vec3 = (0.0, 0.0, 0.0)
    quat_offset: Quat = (1.0, 0.0, 0.0, 0.0)
    fov: float = 60.0
    res: ResHW = (224, 224)
    gui: bool = False
    near: float = 0.1
    far: float = 20.0

    def __post_init__(self) -> None:
        """Validate the camera's field of view and resolution."""
        if not 0.0 < self.fov < 180.0:
            raise ValueError("DynamicCamOptions.fov must be in (0, 180).")
        if min(self.res) <= 0:
            raise ValueError("DynamicCamOptions.res must be positive integers.")
        if not 0.0 < self.near < self.far < float("inf"):
            raise ValueError("Camera clipping requires finite 0 < near < far.")
        if not isinstance(self.gui, bool):
            raise TypeError("DynamicCamOptions.gui must be a bool.")


@dataclass(slots=True)
class CamerasOptions:
    """Configuration for all cameras and rendering modalities in the environment.

    This class manages both fixed (static) and robot-mounted (dynamic) cameras.
    It supports loading preset camera configurations from YAML files. If no
    cameras are specified, a default 'front' and 'wrist' camera are used.

    Note:
        String entries for static cameras are expanded into StaticCamOptions by
        loading presets from cameras.yaml. String entries for dynamic
        cameras are resolved by CameraSystem after the robot has been built.

    Attributes:
        static: A dictionary mapping camera names to StaticCamOptions or
            preset string keys. Defaults to a single 'front' camera.
        dynamic: A dictionary mapping camera names to DynamicCamOptions or
            preset string keys. Defaults to a single 'wrist' camera.
        modalities: A tuple of rendering modalities (e.g., 'rgb', 'depth') to
            generate for all cameras. 'rgb' is required.
    """

    static: dict[str, StaticCamOptions | str] = field(
        default_factory=lambda: {"front": "front"},
    )
    dynamic: dict[str, DynamicCamOptions | str] = field(
        default_factory=lambda: {"wrist": "wrist_cam"},
    )
    modalities: tuple[str, ...] = ("rgb",)

    def __post_init__(self) -> None:
        """Validate modalities and expands camera presets from YAML."""
        self._validate_modalities()
        object.__setattr__(self, "static", self._expand_static_presets())
        self._validate_dynamic_entries()

    def _validate_modalities(self) -> None:
        if not isinstance(self.modalities, tuple):
            raise TypeError("CamerasOptions.modalities must be a tuple of strings.")
        for m in self.modalities:
            if m not in _RENDER_MODALITIES:
                raise ValueError(
                    f"Unknown modality '{m}'. Allowed: {_RENDER_MODALITIES}",
                )
        if "rgb" not in self.modalities:
            raise ValueError("CamerasOptions.modalities must include 'rgb'.")

    def _expand_static_presets(self) -> dict[str, StaticCamOptions]:
        cfg = load_config("cameras")
        if not isinstance(cfg, dict):
            raise TypeError("config/cameras.yaml must be a mapping.")
        static_cfg = cfg.get("static")
        if not isinstance(static_cfg, dict):
            raise TypeError(
                "'static' section missing or not a mapping in cameras.yaml.",
            )

        expanded_static: dict[str, StaticCamOptions] = {}
        for name, entry in list(self.static.items()):
            if isinstance(entry, str):
                if entry not in static_cfg:
                    raise ValueError(
                        f"Static preset '{entry}' not found under 'static' "
                        "in cameras.yaml.",
                    )
                preset = static_cfg[entry]
                if not isinstance(preset, dict):
                    raise TypeError(
                        f"Static preset '{entry}' under 'static' in cameras.yaml "
                        "must be a mapping.",
                    )
                for k in ("pos", "lookat", "fov", "res"):
                    if k not in preset:
                        raise ValueError(
                            f"Static preset '{entry}' is missing key '{k}'.",
                        )
                expanded_static[name] = StaticCamOptions(
                    pos=tuple(preset["pos"]),
                    lookat=tuple(preset["lookat"]),
                    fov=float(preset["fov"]),
                    res=tuple(preset["res"]),
                    gui=preset.get("gui", False),
                    near=float(preset.get("near", 0.1)),
                    far=float(preset.get("far", 20.0)),
                )
            elif isinstance(entry, StaticCamOptions):
                expanded_static[name] = entry
            else:
                raise TypeError(
                    f"CamerasOptions.static['{name}'] must be str or StaticCamOptions.",
                )
        return expanded_static

    def _validate_dynamic_entries(self) -> None:
        for name, entry in list(self.dynamic.items()):
            if isinstance(entry, (DynamicCamOptions, str)):
                continue
            raise TypeError(
                f"CamerasOptions.dynamic['{name}'] must be str or DynamicCamOptions.",
            )
