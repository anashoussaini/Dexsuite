"""Define the Robotiq 2F-85 gripper model."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("robotiq")
class Robotiq85(GripperModel):
    """Represent the Robotiq 2F-85 gripper."""

    model_path: str = package_asset_path(__package__, "2f85.xml")
    dof: int = 1

    root_link: str = "base"
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, 0.0, 0.16, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    q_lo: torch.Tensor = torch.tensor([0.0], dtype=torch.float32)
    q_hi: torch.Tensor = torch.tensor([0.85], dtype=torch.float32)
    open_q: torch.Tensor = torch.tensor([0.0], dtype=torch.float32)
    closed_q: torch.Tensor = torch.tensor([0.85], dtype=torch.float32)
    home_q: torch.Tensor = open_q.clone()
