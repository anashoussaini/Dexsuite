"""Define the arm-less floating wrist manipulator model.

A virtual 6-DoF gantry (x, y, z slides followed by roll, pitch, yaw hinges) with
no geometry: the hand attached to its ``attachment`` link floats in free space
and its 6-DoF pose is commanded directly, as with a handheld data-collection rig.
"""

from __future__ import annotations

import math

import torch

from dexsuite.core.components import ModularManipulatorModel
from dexsuite.core.registry import register_manipulator
from dexsuite.utils.models_utils import package_asset_path


@register_manipulator("floating")
class FloatingWrist(ModularManipulatorModel):
    """Represent a directly commanded 6-DoF wrist with no arm."""

    model_path: str = package_asset_path(__package__, "floating_wrist.xml")
    dof: int = 6

    root_link: str = "gantry_x"
    end_link: str = "attachment"

    # Hand root exactly where the Franka + elbow-mount home puts it (0.377, 0, 0.520); hinges at zero so the
    # roll-pitch-yaw singularity (pitch = +-90 deg, fingers vertical) is far from the working pose.
    home_q: torch.Tensor = torch.tensor(
        [0.377, 0.0, 0.5203, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    q_lo: torch.Tensor = torch.tensor(
        [-0.5, -1.0, -0.1, -math.pi, -math.pi, -math.pi],
        dtype=torch.float32,
    )
    q_hi: torch.Tensor = torch.tensor(
        [1.5, 1.0, 1.2, math.pi, math.pi, math.pi],
        dtype=torch.float32,
    )

    # Stiff human-arm-like wrist: slides in N/m, N s/m, N; hinges in N m/rad, N m s/rad, N m.
    kp: torch.Tensor = torch.tensor([3000.0] * 3 + [300.0] * 3, dtype=torch.float32)
    kv: torch.Tensor = torch.tensor([150.0] * 3 + [20.0] * 3, dtype=torch.float32)
    force_range: torch.Tensor = torch.tensor([600.0] * 3 + [60.0] * 3, dtype=torch.float32)
