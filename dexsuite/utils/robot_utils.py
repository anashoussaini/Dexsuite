"""Robot-facing utility functions (Genesis integration).

This module contains helpers that depend on Genesis (genesis), such as morph
construction and querying link poses. Keep imports from this module explicit to
avoid pulling heavy dependencies into lightweight contexts.
"""

from __future__ import annotations

import genesis as gs
import torch

from dexsuite.core.components.base import RigidBodyModel
from dexsuite.core.components.mount import GripperMount
from dexsuite.utils.globals import get_device
from dexsuite.utils.orientation_utils import (
    quat_mul_wxyz_torch,
    quat_to_rotmat_wxyz_torch,
    rpy_to_quat_wxyz_torch,
)


def _to_constructor_value(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=torch.float32).tolist()
    return x


def create_mount_morph(
    mount: GripperMount,
) -> gs.morphs.Cylinder | gs.morphs.Box | None:
    """Create a Genesis morphology for a gripper mount.

    Based on the mount's type, this function constructs either a Cylinder
    or a Box morphology. If the mount type is 'invisible', it returns None.

    Args:
        mount: The GripperMount configuration object.

    Returns:
        A Genesis Cylinder or Box morph, or None if the mount is invisible.

    Raises:
        ValueError: If the mount_type is unknown.
    """
    if mount.mount_type == "invisible":
        return None

    pos_value = _to_constructor_value(mount.mount_pos)
    quat_value = _to_constructor_value(mount.mount_quat)

    if mount.mount_type == "cylinder":
        visualization = bool(mount.geometry.get("visualization", True))
        collision = bool(mount.geometry.get("collision", True))
        fixed = bool(mount.geometry.get("fixed", True))
        return gs.morphs.Cylinder(
            pos=pos_value,
            quat=quat_value,
            radius=mount.geometry["radius"],
            height=mount.geometry["height"],
            visualization=visualization,
            collision=collision,
            fixed=fixed,
            batch_fixed_verts=True,
        )

    if mount.mount_type == "box":
        sx, sy, sz = mount.geometry["size"]
        visualization = bool(mount.geometry.get("visualization", True))
        collision = bool(mount.geometry.get("collision", True))
        fixed = bool(mount.geometry.get("fixed", True))
        return gs.morphs.Box(
            pos=pos_value,
            quat=quat_value,
            size=(sx, sy, sz),
            visualization=visualization,
            collision=collision,
            fixed=fixed,
            batch_fixed_verts=True,
        )
    raise ValueError(f"Unknown mount type: '{mount.mount_type}'")


def rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """Convert roll, pitch, and yaw angles to a WXYZ quaternion.

    Args:
        rpy: A tensor of shape (..., 3) representing [roll, pitch, yaw] in radians.

    Returns:
        A tensor of shape (..., 4) representing quaternions in (w, x, y, z)
            order.
    """
    return rpy_to_quat_wxyz_torch(rpy).to(get_device(), dtype=torch.float32)


def get_robot_morph(
    robot_model: RigidBodyModel,
    **kwargs,
) -> gs.morphs.URDF | gs.morphs.MJCF:
    """Create a Genesis morphology from a robot model.

    Args:
        robot_model: The RigidBodyModel instance defining the robot.
        kwargs: Additional keyword arguments passed to the morphology constructor (for example, base position pos, orientation quat).

    Returns:
        A Genesis MJCF or URDF morph instance.

    Raises:
        ValueError: If the model file extension is not supported (.xml or .urdf).
    """
    if "pos" in kwargs and isinstance(kwargs["pos"], torch.Tensor):
        kwargs["pos"] = _to_constructor_value(kwargs["pos"])
    if "quat" in kwargs and isinstance(kwargs["quat"], torch.Tensor):
        kwargs["quat"] = _to_constructor_value(kwargs["quat"])

    robot_morph = None
    if robot_model.model_path.endswith(".xml"):
        batch_fixed_verts = getattr(robot_model, "batch_fixed_verts", None)
        if batch_fixed_verts is not None and "batch_fixed_verts" not in kwargs:
            kwargs["batch_fixed_verts"] = bool(batch_fixed_verts)
        robot_morph = gs.morphs.MJCF(file=robot_model.model_path, **kwargs)
    elif robot_model.model_path.endswith(".urdf") or robot_model.model_path.endswith(
        ".URDF",
    ):
        merge_fixed_links = getattr(robot_model, "merge_fixed_links", True)
        links_to_keep = getattr(robot_model, "links_to_keep", ())
        recompute_inertia = getattr(robot_model, "recompute_inertia", False)
        fixed_base = getattr(robot_model, "fixed_base", True)
        robot_morph = gs.morphs.URDF(
            file=robot_model.model_path,
            fixed=fixed_base,
            merge_fixed_links=merge_fixed_links,
            links_to_keep=links_to_keep,
            recompute_inertia=recompute_inertia,
            **kwargs,
        )
    else:
        raise ValueError("The model file is not supported")
    return robot_morph


def link_entities(
    parent_entity,
    child_entity,
    parent_link_name: str = "",
    child_link_name: str = "",
) -> None:
    """Attach one Genesis entity to another before the scene is built.

    Args:
        parent_entity: Entity providing the parent link.
        child_entity: Entity whose baselink will be attached.
        parent_link_name: Optional parent-link name. Empty selects the entity default.
        child_link_name: Optional child-link name. Must resolve to the child baselink.

    Raises:
        RuntimeError: If either entity is already built, the child cannot be
            attached, or Genesis attach fails.
    """
    if getattr(parent_entity, "is_built", False) or getattr(child_entity, "is_built", False):
        raise RuntimeError("link_entities only supports pre-build Genesis attach().")

    if not hasattr(child_entity, "attach"):
        raise RuntimeError("Child entity does not expose Genesis attach().")

    parent_link = parent_entity.get_link(parent_link_name)
    base_link = child_entity.links[0]
    requested_child = base_link if not child_link_name else child_entity.get_link(child_link_name)

    if requested_child.idx != base_link.idx:
        raise RuntimeError(
            "Child link must be the entity baselink for Genesis attach(). "
            f"requested={requested_child.name!r}, baselink={base_link.name!r}.",
        )

    try:
        child_entity.attach(
            parent_entity,
            parent_link_name=parent_link_name or None,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Genesis attach failed while linking entities. "
            f"parent_link={parent_link.name!r}, child_link={requested_child.name!r}.",
        ) from exc

def get_end_effector_pose(
    entity,
    link_idx: int,
    qpos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward kinematics for a given joint configuration.

    Args:
        entity: Genesis entity implementing forward_kinematics.
        link_idx: Link index local to entity.
        qpos: Joint positions with shape (dof,) or (B, dof).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (pos, quat) with shapes (3,) or (B, 3)
        and (4,) or (B, 4) respectively.
    """
    pos, quat = entity.forward_kinematics(
        qpos=qpos.contiguous(),
        links_idx_local=[link_idx],
    )
    pos = pos.contiguous()
    quat = quat.contiguous()
    return pos, quat


def get_tool_center_point_pose(
    robot,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the world pose of the tool center point.

    This is computed from the gripper root-link pose and the static local TCP
    offset stored on the robot. No fake rigid anchor is created.

    Args:
        robot: The robot instance (e.g., IntegratedRobot, ModularRobot).

    Returns:
        A tuple containing:
            - pos: Position tensor of shape (batch_size, 3).
            - quat: Quaternion tensor (w, x, y, z) of shape (batch_size, 4).
    """
    link = robot.hand_ent.get_link(robot.gripper.root_link)
    pos = link.get_pos()
    quat = link.get_quat()

    tcp_local_pos = getattr(robot, "_tcp_local_pos", None)
    tcp_local_quat = getattr(robot, "_tcp_local_quat", None)
    if tcp_local_pos is None or tcp_local_quat is None:
        return pos, quat

    world_offset = torch.matmul(
        quat_to_rotmat_wxyz_torch(quat),
        tcp_local_pos.unsqueeze(-1),
    ).squeeze(-1)
    tcp_pos = pos + world_offset
    tcp_quat = quat_mul_wxyz_torch(quat, tcp_local_quat)
    return tcp_pos, tcp_quat


def attach_tcp_anchor(robot, visualize: bool = True) -> None:
    """Stores the gripper-local TCP pose on the robot.

    Args:
        robot: The robot instance. Expected attributes: gripper and device.
        visualize: Unused for physics. Kept for API compatibility.
    """
    del visualize
    tcp_pose = getattr(robot.gripper, "tcp_pose", None)
    if tcp_pose is None:
        robot._tcp_local_pos = None
        robot._tcp_local_quat = None
        return

    dev = robot.device
    tcp = torch.as_tensor(tcp_pose, dtype=torch.float32, device=dev)
    robot._tcp_local_pos = tcp[:3].contiguous()
    robot._tcp_local_quat = rpy_to_quat(tcp[3:]).contiguous()
