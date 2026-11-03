"""Define the Kinova Gen3 manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("gen3")
class Gen3(ModularManipulatorModel):
    """Represent the Kinova Gen3 arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "gen3.xml")
    dof: int = 7
    root_link: str = "base_link"
    end_link: str = "bracelet_link"
    home_q: torch.Tensor = torch.tensor(
        [0.0, 0.0, 0.0, 1.5708, 0.0, 1.5708, -1.5708],
        dtype=torch.float32,
    )
