"""Define the Ability Hand gripper model.

This module registers the right-handed Ability Hand as a DexSuite gripper.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("ability")
class Ability(GripperModel):
    """Represent the right Ability Hand gripper."""

    model_path: str = package_asset_path(__package__, "ability_hand_right.urdf")
    dof: int = 6
    root_link: str = "base_link"
    home_q: torch.Tensor = torch.zeros((6,), dtype=torch.float32)
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5]
