"""Define the Schunk SVH hand gripper model."""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("schunk")
class Schunk(GripperModel):
    """Represent the right Schunk SVH hand gripper."""

    model_path: str = package_asset_path(__package__, "schunk_svh_hand_right.urdf")
    recompute_inertia: bool = True
    dof: int = 9
    root_link: str = "base_link"
    home_q: torch.Tensor = torch.zeros((9,), dtype=torch.float32)
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    control_dofs_index: ClassVar[list[int]] = [1, 3, 5, 6, 7, 8, 9, 10, 12]
