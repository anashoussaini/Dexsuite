"""Define the Unitree Z1 manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("z1")
class Z1(ModularManipulatorModel):
    """Represent the Unitree Z1 arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "z1.xml")
    dof: int = 6
    root_link: str = "link00"
    end_link: str = "link06"
    # Tool straight down, TCP about 0.3 m out and 0.16 m up; the last joint turns
    # the hand to face away from the base, as on the Franka (same wrist view).
    home_q: torch.Tensor = torch.tensor(
        [0.0, 1.8, -1.6, 1.4, 0.0, 1.5708],
        dtype=torch.float32,
    )
