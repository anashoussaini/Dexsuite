"""Define the BarrettHand gripper model."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("barrett")
class Barrett(GripperModel):
    """Represent the BarrettHand gripper."""

    model_path: str = package_asset_path(__package__, "bhand_model.urdf")
    dof: int = 8
    root_link: str = "base_link"
    home_q: torch.Tensor = torch.zeros((8,), dtype=torch.float32)
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
