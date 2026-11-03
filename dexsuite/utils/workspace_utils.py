"""Compute world-frame axis-aligned robot workspaces."""

from __future__ import annotations

from collections.abc import Sequence

import genesis as gs
import torch

from dexsuite.options import ArmOptions, PoseOptions, RobotOptions
from dexsuite.utils.aabb import AABB
from dexsuite.utils.orientation_utils import rpy_to_quat

Vec3 = Sequence[float] | torch.Tensor


def _aabb_corners(min_pt: Vec3, max_pt: Vec3) -> torch.Tensor:
    """Return the 8 CPU corner points of an axis-aligned box."""
    def _to_tensor3(x: Vec3) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device="cpu").reshape(3)

    lo = _to_tensor3(min_pt)
    hi = _to_tensor3(max_pt)

    xs = [lo[0], hi[0]]
    ys = [lo[1], hi[1]]
    zs = [lo[2], hi[2]]

    return torch.tensor(
        [
            [xs[0], ys[0], zs[0]],
            [xs[1], ys[0], zs[0]],
            [xs[0], ys[1], zs[0]],
            [xs[1], ys[1], zs[0]],
            [xs[0], ys[0], zs[1]],
            [xs[1], ys[0], zs[1]],
            [xs[0], ys[1], zs[1]],
            [xs[1], ys[1], zs[1]],
        ],
        dtype=torch.float32,
    )


def _add_corner_markers(
    scene: gs.Scene,
    corners: torch.Tensor | Sequence[Sequence[float]],
    radius: float = 0.01,
) -> list:
    """Add fixed, non-colliding sphere markers at corner points."""
    corners_t = torch.as_tensor(corners, dtype=torch.float32, device="cpu").reshape(
        -1,
        3,
    )
    ents = []
    for p in corners_t:
        morph = gs.morphs.Sphere(
            pos=(float(p[0]), float(p[1]), float(p[2])),
            radius=radius,
            fixed=True,
            visualization=True,
            collision=False,
        )
        ent = scene.add_entity(
            morph,
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        ents.append(ent)
    return ents


def draw_workspace_corners(
    scene: gs.Scene,
    aabb: AABB | None,
    radius: float = 0.01,
) -> list:
    """Draw visual sphere markers at world-frame AABB corners.

    Args:
        scene: Genesis scene that receives markers.
        aabb: Axis-aligned bounding box to draw.
        radius: Corner marker radius. Defaults to 0.01.
    """
    if aabb is None:
        return []
    corners = _aabb_corners(aabb.min, aabb.max)
    return _add_corner_markers(scene, corners, radius=radius)


def _pose_quat_wxyz(pose: PoseOptions | None) -> tuple[float, float, float, float]:
    """Return the base quaternion as WXYZ."""
    if pose is None:
        return (1.0, 0.0, 0.0, 0.0)

    if pose.quat is not None:
        w, x, y, z = pose.quat
        return (float(w), float(x), float(y), float(z))

    q = rpy_to_quat(torch.tensor([0.0, 0.0, float(pose.yaw_rad)]))
    return tuple(map(float, q.tolist()))


def _arm_world_aabb(
    arm: ArmOptions | None,
    pose: PoseOptions | None,
    device: torch.device,
) -> AABB | None:
    """Compute the world-frame AABB for a single arm, if defined."""
    if arm is None or arm.workspace is None or pose is None:
        return None

    local = AABB.from_lists(arm.workspace.min, arm.workspace.max)
    world = local.transform_by_base(
        base_pos=pose.pos,
        base_quat_wxyz=_pose_quat_wxyz(pose),
    )
    return AABB.from_tensors(world.min.to(device), world.max.to(device))


def compute_world_aabb_from_options(
    ro: RobotOptions,
    device: torch.device,
) -> dict[str, AABB | None]:
    """Build world-frame workspaces (union + per-arm) for the robot.

        Returns a dict with keys: "union", "left", "right", "overlap".
        For single-arm robots, left/right/overlap are None and union is the
        single workspace. For bimanual robots, union is the box covering both
        arms, and overlap is the intersection (or None if disjoint).

    Args:
        ro: Robot options used to compute workspaces.
        device: Torch device used for tensors.
    """
    if ro.type_of_robot == "bimanual":
        return _compute_bimanual_world_aabbs(ro, device)

    single = _arm_world_aabb(ro.single, ro.layout.single, device)
    return {"union": single, "left": None, "right": None, "overlap": None}


def _compute_bimanual_world_aabbs(
    ro: RobotOptions,
    device: torch.device,
) -> dict[str, AABB | None]:
    """Return per-arm, overlap, and union AABBs."""
    if ro.type_of_robot != "bimanual":
        single = _arm_world_aabb(ro.single, ro.layout.single, device)
        return {"left": None, "right": None, "overlap": None, "union": single}

    left = _arm_world_aabb(ro.left, ro.layout.left, device)
    right = _arm_world_aabb(ro.right, ro.layout.right, device)

    if left is None and right is None:
        return {"left": None, "right": None, "overlap": None, "union": None}
    if left is None:
        return {"left": None, "right": right, "overlap": None, "union": right}
    if right is None:
        return {"left": left, "right": None, "overlap": None, "union": left}

    union = left.union(right)
    overlap = left.intersection(right)
    return {"left": left, "right": right, "overlap": overlap, "union": union}


def outside_aabb(
    pos: torch.Tensor,
    aabb_min: torch.Tensor | Sequence[float],
    aabb_max: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Return whether positions are outside tensor AABB bounds.

    Args:
        pos: Position tensor with shape (..., 3).
        aabb_min: Minimum AABB boundary with shape (3,).
        aabb_max: Maximum AABB boundary with shape (3,).

    Returns:
        Boolean tensor with shape `pos.shape[:-1]`.
    """
    lo = torch.as_tensor(aabb_min, dtype=pos.dtype, device=pos.device)
    hi = torch.as_tensor(aabb_max, dtype=pos.dtype, device=pos.device)
    return torch.any((pos < lo) | (pos > hi), dim=-1)
