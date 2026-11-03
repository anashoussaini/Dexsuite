"""Define the DClaw gripper model."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("dclaw")
class Dclaw(GripperModel):
    """Represent the DClaw gripper."""

    model_path: str = package_asset_path(__package__, "dclaw_gripper_glb.urdf")
    dof: int = 9
    root_link: str = "base"
    home_q: torch.Tensor = torch.zeros((9,), dtype=torch.float32)
    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, -0.035, 0.09, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
