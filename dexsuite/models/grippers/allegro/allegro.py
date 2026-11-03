"""Define left and right Allegro Hand gripper models."""

from __future__ import annotations

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path


class _AllegroBase(GripperModel):
    """Provide shared Allegro hand settings."""

    dof: int = 16
    root_link: str = "palm"
    tcp_pose: torch.Tensor = torch.tensor(
        [0.06, 0.0, 0.02, 0.0, torch.pi, torch.pi / 2],
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
        ],
        dtype=torch.float32,
    )


@register_gripper("allegro_right")
@register_gripper("allegro")
class AllegroRight(_AllegroBase):
    """Represent the right Allegro Hand gripper."""

    model_path = package_asset_path(__package__, "right_hand.xml")


@register_gripper("allegro_left")
class AllegroLeft(_AllegroBase):
    """Represent the left Allegro Hand gripper."""

    model_path = package_asset_path(__package__, "left_hand.xml")
