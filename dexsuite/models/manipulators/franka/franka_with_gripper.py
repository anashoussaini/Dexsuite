"""Franka Emika Panda model with an integrated 2F gripper.

This module provides a single, integrated manipulator model that includes both
the 7-DoF Panda arm and its 2-DoF parallel gripper in one MJCF file.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("franka_with_gripper")
class FrankaWithGripper(IntegratedManipulatorModel):
    """Represent the Franka Emika Panda manipulator with its built-in gripper.

    The total DoF count includes the arm and parallel gripper, while `home_q`
    contains the arm-only home pose selected by `control_dofs_index`. The
    gripper joints use explicit full-entity DoF indices.
    """

    model_path: str = package_asset_path(__package__, "panda.xml")
    dof: int = 9

    root_link: str = "link0"
    end_link: str = "link7"

    control_dofs_index: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5, 6]
    home_q: torch.Tensor = torch.tensor(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, -0.785],
        dtype=torch.float32,
    )
    builtin_gripper_dof: int = 2
    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [7, 8]
    builtin_gripper_root_link: str = "hand"
    builtin_gripper_q_lo: torch.Tensor = torch.tensor([0.0, 0.0], dtype=torch.float32)
    builtin_gripper_q_hi: torch.Tensor = torch.tensor(
        [0.04, 0.04],
        dtype=torch.float32,
    )
    builtin_gripper_home_q: torch.Tensor = torch.tensor(
        [0.04, 0.04],
        dtype=torch.float32,
    )
    builtin_gripper_tcp_pose: torch.Tensor | None = None
