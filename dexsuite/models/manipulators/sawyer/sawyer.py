"""Define the Rethink Robotics Sawyer manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("sawyer")
class Sawyer(ModularManipulatorModel):
    """Represent the Rethink Robotics Sawyer arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "sawyer.xml")
    dof: int = 7
    root_link: str = "base"
    end_link: str = "right_l6"
    # Tool straight down, TCP about 0.5 m out and 0.19 m up; the last joint turns
    # the hand to face away from the base, as on the Franka (same wrist view).
    home_q: torch.Tensor = torch.tensor(
        [0.0, -1.4, 0.0, 1.95, 0.0, 1.0, 1.7453],
        dtype=torch.float32,
    )
