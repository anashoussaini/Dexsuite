"""Joint helpers for articulated props (URDF entities with revolute/prismatic joints).

All functions address one joint by name and work for a single env (``envs_idx=None``) or a
subset of a batched scene. Positions are returned with a trailing joint axis, ``(1,)`` for a
single env and ``(B, 1)`` batched, so they drop straight into ``_get_extra_obs``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def joint_dof(entity, joint_name: str) -> int:
    """Return the entity-local dof index of a 1-dof joint."""
    joint = entity.get_joint(joint_name)
    dofs = list(joint.dofs_idx_local)
    if len(dofs) != 1:
        raise ValueError(f"joint {joint_name!r} has {len(dofs)} dofs; expected 1")
    return int(dofs[0])


def joint_q(entity, joint_name: str, envs_idx=None) -> torch.Tensor:
    """Joint position, shape ``(1,)`` (single env) or ``(B, 1)``."""
    return entity.get_dofs_position(dofs_idx_local=[joint_dof(entity, joint_name)], envs_idx=envs_idx)


def joint_qd(entity, joint_name: str, envs_idx=None) -> torch.Tensor:
    """Joint velocity, shape ``(1,)`` (single env) or ``(B, 1)``."""
    return entity.get_dofs_velocity(dofs_idx_local=[joint_dof(entity, joint_name)], envs_idx=envs_idx)


def set_joint_q(entity, joint_name: str, q: float | torch.Tensor, envs_idx=None, *, zero_velocity: bool = True) -> None:
    """Set one joint's position (and by default zero every velocity of the entity)."""
    dof = [joint_dof(entity, joint_name)]
    value = torch.as_tensor(q, dtype=torch.float32).reshape(-1, 1)
    n_envs = int(entity.solver.n_envs)
    if n_envs == 0:
        value = value.reshape(1)
    else:
        n = n_envs if envs_idx is None else int(torch.as_tensor(envs_idx).numel())
        value = value.expand(n, 1) if value.shape[0] == 1 else value
    entity.set_dofs_position(value, dofs_idx_local=dof, envs_idx=envs_idx, zero_velocity=zero_velocity)


def set_joint_dynamics(
    entity,
    joint_name: str,
    *,
    damping: float | None = None,
    friction: float | None = None,
    stiffness: float | None = None,
    armature: float | None = None,
) -> None:
    """Post-build passive dynamics for one joint (values in SI, per Genesis conventions)."""
    dof = [joint_dof(entity, joint_name)]
    if damping is not None:
        entity.set_dofs_damping(torch.tensor([float(damping)]), dofs_idx_local=dof)
    if friction is not None:
        entity.set_dofs_frictionloss(torch.tensor([float(friction)]), dofs_idx_local=dof)
    if stiffness is not None:
        entity.set_dofs_stiffness(torch.tensor([float(stiffness)]), dofs_idx_local=dof)
    if armature is not None:
        entity.set_dofs_armature(torch.tensor([float(armature)]), dofs_idx_local=dof)


def set_joint_force(entity, joint_name: str, force: float, envs_idx=None) -> None:
    """Apply a motor force/torque on one joint (clamped to the URDF effort).

    Genesis keeps the control force on every substep until it is overwritten or ``scene.reset()``
    runs; call :func:`clear_joint_force` to release it.
    """
    dof = [joint_dof(entity, joint_name)]
    value = torch.tensor([float(force)])
    n_envs = int(entity.solver.n_envs)
    if n_envs > 0:
        n = n_envs if envs_idx is None else int(torch.as_tensor(envs_idx).numel())
        value = value.reshape(1, 1).expand(n, 1)
    entity.control_dofs_force(value, dofs_idx_local=dof, envs_idx=envs_idx)


def clear_joint_force(entity, joint_name: str, envs_idx=None) -> None:
    """Release a force applied with :func:`set_joint_force`."""
    set_joint_force(entity, joint_name, 0.0, envs_idx)


def quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors ``v`` into the frame with orientation ``q`` (wxyz); broadcasts over leading dims."""
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    qv = torch.cat([x, y, z], dim=-1)
    # v' = v - 2w (qv x v) + 2 (qv x (qv x v))
    c1 = torch.cross(qv, v, dim=-1)
    c2 = torch.cross(qv, c1, dim=-1)
    return v - 2.0 * w * c1 + 2.0 * c2


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate frame vectors ``v`` into the world with orientation ``q`` (wxyz)."""
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    qv = torch.cat([x, y, z], dim=-1)
    c1 = torch.cross(qv, v, dim=-1)
    c2 = torch.cross(qv, c1, dim=-1)
    return v + 2.0 * w * c1 + 2.0 * c2


def point_in_box(p_local: torch.Tensor, lo: Sequence[float], hi: Sequence[float]) -> torch.Tensor:
    """Elementwise AABB containment of local points ``(..., 3)`` -> bool ``(...)``."""
    lo_t = torch.as_tensor(lo, dtype=p_local.dtype, device=p_local.device)
    hi_t = torch.as_tensor(hi, dtype=p_local.dtype, device=p_local.device)
    return ((p_local >= lo_t) & (p_local <= hi_t)).all(dim=-1)
