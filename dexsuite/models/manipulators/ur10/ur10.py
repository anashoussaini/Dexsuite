"""Define the Universal Robots UR10e manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("ur10")
class Ur10(ModularManipulatorModel):
    """Represent the Universal Robots UR10e arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "ur10e.xml")
    dof: int = 6

    root_link: str = "base"
    end_link: str = "attachment"

    home_q: torch.Tensor = torch.tensor(
        [0, -1.5708, -1.5708, -1.5708, 1.5708, 0.0],
        dtype=torch.float32,
    )
