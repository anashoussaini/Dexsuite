"""UMI parallel-jaw gripper with tactile fingertip pads."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


@register_gripper("umi")
@register_gripper("franka_umi_tacticle")  # the name the recorded UMI datasets use: the same gripper
class Umi(GripperModel):
    """UMI parallel-jaw gripper with ALOHA-style fingertips and tactile pads."""

    model_path: str = package_asset_path(__package__, "umi.xml")
    dof: int = 2
    root_link: str = "hand"
    visualize_contact: bool = False

    q_lo: torch.Tensor = torch.tensor([0.0, -0.04], dtype=torch.float32)
    q_hi: torch.Tensor = torch.tensor([0.04, 0.0], dtype=torch.float32)
    home_q: torch.Tensor = torch.tensor([0.04, -0.04], dtype=torch.float32)
    kp: torch.Tensor = torch.tensor([1200.0, 1200.0], dtype=torch.float32)
    kv: torch.Tensor = torch.tensor([150.0, 150.0], dtype=torch.float32)
    force_range: torch.Tensor = torch.tensor([300.0, 300.0], dtype=torch.float32)

    tcp_pose: torch.Tensor = torch.tensor(
        [0.10, 0.0, 0.0584, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    tactile: bool = True
    tactile_links: tuple[str, str] = ("left_elastomer", "right_elastomer")
    tactile_grid_size: tuple[int, int, int] = (12, 1, 32)
    tactile_point_distance: float = 0.002
