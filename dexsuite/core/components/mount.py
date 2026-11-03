"""Define arm-to-gripper mount transforms.

Mount and gripper offsets are stored as float32 torch tensors on the global
DexSuite device. Positions use meters, and quaternions use WXYZ order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dexsuite.utils import get_device


class GripperMount:
    """Represent the pose and geometry of an arm-to-gripper adapter.

    Attributes:
        mount_type: Mount type identifier.
        mount_pos: `(3,)` float32 tensor for mount position relative to the arm flange.
        mount_quat: `(4,)` float32 tensor for mount orientation relative to the arm flange in WXYZ order.
        geometry: Geometry metadata for visual or collision construction.
        gripper_pos: `(3,)` float32 tensor for gripper-root position relative to the mount.
        gripper_quat: `(4,)` float32 tensor for gripper-root orientation relative to the mount in WXYZ order.
        device: Torch device used for mount tensors.
    """

    def __init__(
        self,
        mount_type: str = "invisible",
        mount_pos: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
        mount_quat: Sequence[float] | torch.Tensor = (1.0, 0.0, 0.0, 0.0),
        geometry: dict | None = None,
        gripper_pos: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
        gripper_quat: Sequence[float] | torch.Tensor = (1.0, 0.0, 0.0, 0.0),
    ):
        """Initialize the object.

        Args:
        mount_type: Mount type identifier. Defaults to "invisible".
        mount_pos: Mount position `[x, y, z]` relative to the arm flange. Defaults to (0.0, 0.0, 0.0).
        mount_quat: Mount orientation `[w, x, y, z]` relative to the arm flange. Defaults to (1.0, 0.0, 0.0, 0.0).
        geometry: Mount geometry metadata. Defaults to None.
        gripper_pos: Gripper-root position `[x, y, z]` relative to the mount. Defaults to (0.0, 0.0, 0.0).
        gripper_quat: Gripper-root orientation `[w, x, y, z]` relative to the mount. Defaults to (1.0, 0.0, 0.0, 0.0).
        """
        self.device = get_device()
        self.mount_type = mount_type
        self.mount_pos = torch.tensor(
            mount_pos,
            device=self.device,
            dtype=torch.float32,
        )
        self.mount_quat = torch.tensor(
            mount_quat,
            device=self.device,
            dtype=torch.float32,
        )
        self.geometry = geometry or {}

        self.gripper_pos = torch.tensor(
            gripper_pos,
            device=self.device,
            dtype=torch.float32,
        )
        self.gripper_quat = torch.tensor(
            gripper_quat,
            device=self.device,
            dtype=torch.float32,
        )
