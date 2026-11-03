"""Define the Inspire Hand gripper model.

This module registers the right-handed Inspire Hand gripper.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("inspire")
class Inspire(GripperModel):
    """Represent the right Inspire Hand gripper."""

    model_path: str = package_asset_path(__package__, "inspire_hand_right.urdf")
    dof: int = 6
    root_link: str = "base"

    kp: torch.Tensor = torch.full((6,), 50.0, dtype=torch.float32)
    kv: torch.Tensor = torch.full((6,), 12.8, dtype=torch.float32)
    home_q: torch.Tensor = torch.zeros((6,), dtype=torch.float32)

    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5]
