"""Define left and right LEAP Hand gripper models."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("leap_right")
@register_gripper("leap")
class LeapRight(GripperModel):
    """Represent the right LEAP Hand gripper."""

    model_path: str = package_asset_path(__package__, "right_hand.xml")
    dof: int = 16
    root_link: str = "palm"
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    home_q: torch.Tensor = torch.zeros((16,), dtype=torch.float32)


@register_gripper("leap_left")
class LeapLeft(GripperModel):
    """Represent the left LEAP Hand gripper."""

    model_path: str = package_asset_path(__package__, "left_hand.xml")
    dof: int = 16
    root_link: str = "palm"
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    home_q: torch.Tensor = torch.zeros((16,), dtype=torch.float32)
