"""Define the SO-100 manipulator model."""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("so_100")
class SO100(IntegratedManipulatorModel):
    """Represent the SO Arm 100 with rotary single-jaw gripper."""

    model_path: str = package_asset_path(__package__, "so_arm100.xml")
    dof: int = 6

    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4]

    root_link: str = "Base"

    end_link: str = "Moving_Jaw"

    home_q: torch.Tensor = torch.tensor(
        [0.0, -1.57, 1.57, 1.57, -1.57],
        dtype=torch.float32,
    )

    builtin_gripper_dof: int = 1

    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [5]
    builtin_gripper_root_link: str = "Fixed_Jaw"

    builtin_gripper_q_lo: torch.Tensor = torch.tensor([-0.174], dtype=torch.float32)
    builtin_gripper_q_hi: torch.Tensor = torch.tensor([1.75], dtype=torch.float32)
    builtin_gripper_home_q: torch.Tensor = torch.tensor([1.75], dtype=torch.float32)
    builtin_gripper_tcp_pose: torch.Tensor = torch.tensor(
        [0.012, -0.08, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
