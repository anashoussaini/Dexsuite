"""Define the Link6 modular manipulator model.

This module registers a 6-DoF arm model defined in an MJCF file.
"""

from __future__ import annotations

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_resource_path


@register_manipulator("link6")
class Link6(ModularManipulatorModel):
    """Represent the Link6 arm-only manipulator."""

    model_path: str = package_resource_path(__package__, "link6_best.xml")
    dof: int = 6

    root_link: str = "base_link"
    end_link: str = "end_effector_link"
    # Tool straight down; the last joint turns the hand to face away from the base,
    # as on the Franka (same wrist view).
    home_q: torch.Tensor = torch.tensor(
        [0.0, 0.0, 1.571, 0.0, 0.0, 3.1416],
        dtype=torch.float32,
    )
