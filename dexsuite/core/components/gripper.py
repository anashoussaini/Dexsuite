"""Define gripper and hand model metadata.

Gripper poses use meters and radians, with TCP poses represented as
`[x, y, z, roll, pitch, yaw]` relative to `root_link`. Optional tactile metadata
describes links sampled by Genesis contact-force grids.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dexsuite.core.components.base import RigidBodyModel


class GripperModel(RigidBodyModel):
    """Represent a robotic hand or parallel gripper model.

    Attributes:
        root_link: Link name where the gripper attaches to an arm.
        tcp_pose: TCP pose `[x, y, z, roll, pitch, yaw]` relative to `root_link`, or `None`.
        open_q: Open joint configuration as a `(dof,)` float32 tensor, or `None`.
        closed_q: Closed joint configuration as a `(dof,)` float32 tensor, or `None`.
        tactile: Whether the gripper exposes tactile pads.
        tactile_links: Link names to expose as tactile pads, or `None`.
        tactile_grid_size: Genesis force-grid resolution for each tactile pad, or `None`.
        tactile_point_distance: Spacing in meters between tactile sample points, or `None`.
    """

    root_link: str
    fixed_base: bool = False
    tcp_pose: torch.Tensor | None = None
    open_q: torch.Tensor | None = None
    closed_q: torch.Tensor | None = None

    tactile: bool = False
    tactile_links: Sequence[str] | None = None
    tactile_grid_size: tuple[int, int, int] | None = None
    tactile_point_distance: float | None = None
