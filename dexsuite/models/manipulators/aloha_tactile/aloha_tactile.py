"""Define the single-arm ALOHA tactile manipulator model."""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("aloha_tactile")
class AlohaTactile(IntegratedManipulatorModel):
    """Represent a single-arm ALOHA tactile manipulator."""

    model_path: str = package_asset_path(__package__, "aloha_tactile.xml")
    merge_fixed_links: bool = False
    visualize_contact: bool = True
    tactile: bool = True

    # 6 arm joints + 2 independently actuated finger joints.
    dof: int = 8

    root_link: str = "base_link"
    end_link: str = "gripper_link"

    # Match the EasySim ALOHA position-control stiffness closely enough that
    # contact does not shove the arm or fingers around.
    kp: torch.Tensor = torch.tensor([12800.0] * 6 + [800.0] * 2, dtype=torch.float32)
    kv: torch.Tensor = torch.tensor([800.0] * 6 + [100.0] * 2, dtype=torch.float32)
    force_range: torch.Tensor = torch.tensor([800.0] * 8, dtype=torch.float32)

    home_q: torch.Tensor = torch.tensor(
        [0.0, -0.16, 1.15, 0.0, -0.5, 0.0],
        dtype=torch.float32,
    )

    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5]

    builtin_gripper_dof: int = 2
    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [6, 7]
    builtin_gripper_root_link: str = "gripper_link"
    builtin_gripper_home_q: torch.Tensor = torch.tensor(
        [0.057, -0.057],
        dtype=torch.float32,
    )
    builtin_gripper_q_lo: torch.Tensor = torch.tensor(
        [0.021, -0.057],
        dtype=torch.float32,
    )
    builtin_gripper_q_hi: torch.Tensor = torch.tensor(
        [0.057, -0.021],
        dtype=torch.float32,
    )
    builtin_tactile_links: tuple[str, str] = (
        "left_elastomer",
        "right_elastomer",
    )
    builtin_tactile_grid_size: tuple[int, int, int] = (12, 1, 32)
    builtin_tactile_point_distance: float = 0.002
    builtin_gripper_tcp_pose: torch.Tensor = torch.tensor(
        [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
