"""Base interface for teleoperation devices."""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np


class Device(ABC):
    """Base class for devices that generate robot actions.

    The common action format used across DexSuite device backends is a
    dictionary with a few conventional keys:

    - pose: A 6D or 7D pose action (list/array of floats).
    - gripper: Optional gripper action (usually a 1-element list).
    - reset: Optional reset trigger (usually a 1-element list).
    - event: Optional raw event payload.
    - meta: Optional metadata about the device and connection.

    Device implementations may update their action in a background thread
    (keyboard, SpaceMouse) or compute the action on demand (Vive).
    """

    def __init__(self, controller: str, normalized: bool, gripper: bool) -> None:
        self.controller = str(controller)
        self.normalized = bool(normalized)
        self.gripper = bool(gripper)
        self.action: dict[str, Any] = {}

    def get_action(self) -> dict[str, Any]:
        """Return the latest action dictionary.

        If normalized is True and the action contains a pose entry, pose values
        are clipped to the [-1, 1] range.

        Returns:
            The latest action dictionary.
        """
        if self.normalized and self.action.get("pose") is not None:
            pose = np.asarray(self.action["pose"], dtype=float).reshape(-1)
            self.action["pose"] = np.clip(pose, -1.0, 1.0).tolist()
        return self.action
