"""Define left and right Shadow Hand gripper models."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("shadow_right")
@register_gripper("shadow")
class ShadowRight(GripperModel):
    """Represent the right Shadow Hand gripper."""

    model_path: str = package_asset_path(__package__, "right_hand.xml")
    dof: int = 24
    root_link: str = "rh_forearm"
    tcp_pose: torch.Tensor = torch.tensor(
        [0.07, 0.01, 0.285, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    home_q: torch.Tensor = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    q_hi: torch.Tensor = torch.tensor(
        [
            0.0,
            0.0,
            0.15,
            1.00,
            0.25,
            0.80,
            0.70,
            0.0,
            1.20,
            2.5,
            0.0,
            1.20,
            2.5,
            0.0,
            1.20,
            2.5,
            0.0,
            0.0,
            1.20,
            2.5,
            0.0,
            0.0,
            1.20,
            2.5,
        ],
        dtype=torch.float32,
    )

    q_lo: torch.Tensor = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )


@register_gripper("shadow_left")
class ShadowLeft(GripperModel):
    """Represent the left Shadow Hand gripper."""

    model_path: str = package_asset_path(__package__, "left_hand.xml")
    dof: int = 20
    root_link: str = "lh_forearm"
    home_q: torch.Tensor = torch.zeros((20,), dtype=torch.float32)
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
