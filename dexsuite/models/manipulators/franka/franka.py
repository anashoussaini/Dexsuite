"""Define the Franka Emika Panda arm-only manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("franka")
class FrankaArm(ModularManipulatorModel):
    """Represent the Franka Emika Panda arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "panda_nohand.xml")
    dof: int = 7

    root_link: str = "link0"
    end_link: str = "attachment"

    home_q: torch.Tensor = torch.tensor(
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, -0.785],
        dtype=torch.float32,
    )
