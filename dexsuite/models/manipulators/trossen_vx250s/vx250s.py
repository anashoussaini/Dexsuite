"""Define the Trossen WX250S manipulator model."""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("trossen_wx250s")
class TrossenWX250S(IntegratedManipulatorModel):
    """Represent the Trossen WX250S manipulator with integrated gripper."""

    model_path: str = package_asset_path(__package__, "wx250s.xml")
    dof: int = 7
    root_link: str = "wx250s/base_link"
    end_link: str = "wx250s/gripper_link"
    home_q: torch.Tensor = torch.tensor(
        [0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
        dtype=torch.float32,
    )
    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5]
    builtin_gripper_dof: int = 1
    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [6]
    builtin_gripper_root_link: str = "wx250s/gripper_link"
    builtin_gripper_q_lo: torch.Tensor = torch.tensor([0.015], dtype=torch.float32)
    builtin_gripper_q_hi: torch.Tensor = torch.tensor([0.037], dtype=torch.float32)
    builtin_gripper_home_q: torch.Tensor = torch.tensor([0.015], dtype=torch.float32)
    builtin_gripper_tcp_pose: torch.Tensor | None = None
