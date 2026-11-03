"""Create shared pose-controller backends from normalized configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

CostSpec = float | list[float] | tuple[float, ...]


def make_pose_backend(
    *,
    controller_name: str,
    backend: str,
    entity: Any,
    model: Any,
    dofs_idx: list[int],
    n_envs: int,
    device: torch.device,
    home_q: torch.Tensor | None,
    dt: float,
    solver: str | None,
    frame_type: str | None,
    position_cost: CostSpec,
    orientation_cost: CostSpec,
    posture_cost: CostSpec,
    damping_cost: CostSpec,
    lm_damping: float,
    qp_damping: float,
    max_joint_step: float | None,
    configuration_limit_gain: float,
    pos_tol: float,
    rot_tol: float,
    ik_max_iters: int | None = None,
    ik_damping: float | None = None,
    ik_max_step: float | None = None,
) -> Any:
    """Build the selected pose backend from normalized controller config.

    Args:
        controller_name: Controller name used in validation error messages.
        backend: Pose backend name or alias.
        entity: Genesis entity used by the controller or backend.
        model: Robot component model associated with the entity.
        dofs_idx: Controlled DoF indices in the entity.
        n_envs: Number of parallel environments.
        device: Torch device used for tensors.
        home_q: Home joint configuration for controlled DoFs.
        dt: Controller time step in seconds.
        solver: QP solver name.
        frame_type: Backend task frame type.
        position_cost: Backend position task cost.
        orientation_cost: Backend orientation task cost.
        posture_cost: Backend posture task cost.
        damping_cost: Backend damping task cost.
        lm_damping: Levenberg-Marquardt damping value.
        qp_damping: QP damping value.
        max_joint_step: Maximum joint step per solve.
        configuration_limit_gain: Configuration-limit gain in `(0, 1]`.
        pos_tol: Position tolerance for convergence.
        rot_tol: Rotation tolerance for convergence.
        ik_max_iters: Maximum IK iterations. Defaults to None.
        ik_damping: IK damping value. Defaults to None.
        ik_max_step: Maximum joint step per IK solve. Defaults to None.
    """
    if backend == "genesis_internal":
        from dexsuite.controllers.backends.genesis_internal import (
            make_pose_backend as make_genesis_backend,
        )

        return make_genesis_backend(
            controller_name=controller_name,
            entity=entity,
            dofs_idx=dofs_idx,
            n_envs=n_envs,
            device=device,
            home_q=home_q,
            ik_max_iters=ik_max_iters,
            ik_damping=ik_damping,
            ik_max_step=ik_max_step,
        )

    from dexsuite.controllers.backends.mink import (
        make_pose_backend as make_mink_backend,
    )

    return make_mink_backend(
        controller_name=controller_name,
        model=model,
        dofs_idx=dofs_idx,
        n_envs=n_envs,
        device=device,
        dt=dt,
        solver=solver,
        frame_type=frame_type,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        posture_cost=posture_cost,
        damping_cost=damping_cost,
        lm_damping=lm_damping,
        qp_damping=qp_damping,
        max_joint_step=max_joint_step,
        configuration_limit_gain=configuration_limit_gain,
        pos_tol=pos_tol,
        rot_tol=rot_tol,
    )
