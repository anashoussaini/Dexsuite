"""Define the Kuka LBR iiwa 14 manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("kuka")
class Kuka(ModularManipulatorModel):
    """Represent the Kuka LBR iiwa 14 arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "iiwa14.xml")
    dof: int = 7
    root_link: str = "base"
    end_link: str = "link7"
    # Tool straight down, TCP about 0.6 m out and 0.24 m up; the last joint turns
    # the hand to face away from the base, as on the Franka (same wrist view).
    home_q: torch.Tensor = torch.tensor(
        [0.0, 0.6, 0.0, -1.4, 0.0, 1.15, 1.5708],
        dtype=torch.float32,
    )
