"""Define the UFactory Lite 6 manipulator model."""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("lite6")
class Lite6(ModularManipulatorModel):
    """Represent the UFactory Lite 6 arm-only manipulator."""

    model_path: str = package_asset_path(__package__, "lite6.xml")
    dof: int = 6
    root_link: str = "link_base"
    end_link: str = "link6"
    # Tool straight down, TCP about 0.24 m out and 0.18 m up; the last joint turns
    # the hand to face away from the base, as on the Franka (same wrist view).
    home_q: torch.Tensor = torch.tensor(
        [0.0, 0.0, 1.1, 0.0, 1.1, -1.5708],
        dtype=torch.float32,
    )
