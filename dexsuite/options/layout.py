"""Configuration options for robot placement and workspace layout."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dexsuite.utils.options_utils import load_layout_preset

from .options import Quat, Vec3


def _norm_pose_block(block: dict) -> dict:
    """Normalize a pose dictionary from a YAML configuration.

    This function converts various orientation formats (degrees, radians,
    quaternion) into a consistent format with pos, yaw_rad, and quat.
    Quaternion takes precedence over yaw.

    Args:
        block: A dictionary containing pose information, typically from YAML.
            It must contain a 'pos' key. It may contain 'quat', 'yaw_rad', or
            'yaw_deg'.

    Returns:
        A dictionary with keys 'pos', 'yaw_rad', and 'quat'.

    Raises:
        TypeError: If the input block is not a dictionary.
    """
    if not isinstance(block, dict):
        raise TypeError("Pose block must be a mapping.")
    pos = tuple(block["pos"])
    q = block.get("quat")
    if q is not None:
        return {"pos": pos, "quat": tuple(q), "yaw_rad": 0.0}
    yaw_rad = block.get("yaw_rad")
    yaw_deg = block.get("yaw_deg")
    if yaw_rad is None and yaw_deg is not None:
        yaw_rad = float(yaw_deg) * math.pi / 180.0
    if yaw_rad is None:
        yaw_rad = 0.0
    return {"pos": pos, "yaw_rad": float(yaw_rad), "quat": None}


@dataclass(slots=True)
class PoseOptions:
    """Represent a 6-DoF pose in the world frame.

    Orientation can be specified using either a yaw angle or a quaternion.
    If a quaternion is provided, it takes precedence and yaw_rad must be zero.

    Attributes:
        pos: The (x, y, z) position.
        yaw_rad: The rotation around the z-axis in radians.
        quat: The (w, x, y, z) orientation as a quaternion.
    """

    pos: Vec3 = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    quat: Quat | None = None

    def __post_init__(self) -> None:
        """Validate that orientation is not over-specified."""
        if self.quat is not None and abs(float(self.yaw_rad)) > 1e-12:
            raise ValueError(
                "PoseOptions: provide either quat or yaw_rad (quat takes precedence, yaw_rad must be 0 when quat is set).",
            )


@dataclass(slots=True)
class LayoutOptions:
    """Define the placement of robots in the environment.

    For bimanual setups, a named preset from a configuration file must be
    used to define the poses of the left and right arms. For single-arm
    setups, the pose can be set explicitly.

    Attributes:
        preset: The name of a layout preset for bimanual setups, found in
            the configuration files.
        params: A dictionary of additional parameters for the layout.
        single: The explicit pose for a single-arm setup.
        left: The pose for the left arm in a bimanual setup (resolved from
            preset).
        right: The pose for the right arm in a bimanual setup (resolved from
            preset).
    """

    preset: str | None = None
    params: dict[str, object] = field(default_factory=dict)

    single: PoseOptions | None = None
    left: PoseOptions | None = None
    right: PoseOptions | None = None

    def resolve_bimanual(
        self,
        left_model: str,
        right_model: str,
    ) -> tuple[PoseOptions, PoseOptions]:
        """Load and resolve bimanual layout presets from configuration files.

        Args:
            left_model: The model name of the left manipulator.
            right_model: The model name of the right manipulator.

        Returns:
            A tuple containing the resolved PoseOptions for the left and
            right arms.

        Raises:
            ValueError: If a preset is required but not provided, or if the
                preset is not found in the configuration files for the given
                models.
        """
        if not self.preset:
            raise ValueError("Bimanual layout requires a preset name (string).")
        L = load_layout_preset(left_model)  # noqa: N806
        R = load_layout_preset(right_model)  # noqa: N806
        if self.preset not in L or "left" not in L[self.preset]:
            raise ValueError(
                f"Preset '{self.preset}' not found for '{left_model}' or missing 'left' block.",
            )
        if self.preset not in R or "right" not in R[self.preset]:
            raise ValueError(
                f"Preset '{self.preset}' not found for '{right_model}' or missing 'right' block.",
            )
        left_kwargs = _norm_pose_block(L[self.preset]["left"])
        right_kwargs = _norm_pose_block(R[self.preset]["right"])
        return PoseOptions(**left_kwargs), PoseOptions(**right_kwargs)

    def describe(self) -> str:
        """Return a string description of the layout configuration.

        Returns:
            A descriptive string of the current layout options.
        """
        mode = "bimanual" if (self.left or self.right) else "single"
        return f"LayoutOptions(mode={mode}, preset={self.preset}, params={self.params})"
