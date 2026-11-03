"""Define the Franka Emika FR3 manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("franka_fr3")
class FrankaFR3(ModularManipulatorModel):
    """Represent the Franka Emika FR3 arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "fr3.xml")
    dof: int = 7
    root_link: str = "fr3_link0"
    end_link: str = "fr3_link7"
    home_q: torch.Tensor = torch.tensor(
        [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
        dtype=torch.float32,
    )
