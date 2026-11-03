"""Provide the Genesis internal IK backend for pose controllers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from dexsuite.utils.orientation_utils import (
    quat_mul_wxyz_torch,
    quat_normalize_torch,
    rpy_to_quat_wxyz_torch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _validate_ik_config(
    *,
    controller_name: str,
    ik_max_iters: int | None,
    ik_damping: float | None,
    ik_max_step: float | None,
) -> None:
    if ik_max_iters is not None and not (
        isinstance(ik_max_iters, int) and ik_max_iters > 0
    ):
        raise ValueError(
            f"{controller_name}: ik_max_iters must be > 0 int; "
            f"got {ik_max_iters!r}.",
        )
    if ik_damping is not None and not (
        isinstance(ik_damping, int | float)
        and math.isfinite(float(ik_damping))
        and ik_damping >= 0.0
    ):
        raise ValueError(
            f"{controller_name}: ik_damping must be >= 0, got {ik_damping!r}.",
        )
    if ik_max_step is not None and not (
        isinstance(ik_max_step, int | float)
        and math.isfinite(float(ik_max_step))
        and ik_max_step > 0.0
    ):
        raise ValueError(
            f"{controller_name}: ik_max_step must be > 0 or None, "
            f"got {ik_max_step!r}.",
        )


def make_pose_backend(
    *,
    controller_name: str,
    entity: Any,
    dofs_idx: list[int],
    n_envs: int,
    device: torch.device,
    home_q: torch.Tensor | None = None,
    ik_max_iters: int | None = None,
    ik_damping: float | None = None,
    ik_max_step: float | None = None,
) -> GenesisInternalPoseBackend:
    """Build a Genesis internal IK pose backend from controller config.

    Args:
        controller_name: Controller name used in validation error messages.
        entity: Genesis entity used by the controller or backend.
        dofs_idx: Controlled DoF indices in the entity.
        n_envs: Number of parallel environments.
        device: Torch device used for tensors.
        home_q: Home joint configuration for controlled DoFs. Defaults to None.
        ik_max_iters: Maximum IK iterations. Defaults to None.
        ik_damping: IK damping value. Defaults to None.
        ik_max_step: Maximum joint step per IK solve. Defaults to None.
    """
    return GenesisInternalPoseBackend(
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


class GenesisInternalPoseBackend:
    """Pose backend backed by ``entity.inverse_kinematics``."""

    def __init__(
        self,
        *,
        controller_name: str,
        entity: Any,
        dofs_idx: list[int],
        n_envs: int,
        device: torch.device,
        home_q: torch.Tensor | None = None,
        ik_max_iters: int | None = None,
        ik_damping: float | None = None,
        ik_max_step: float | None = None,
    ) -> None:
        """Initialize the object.

        Args:
        controller_name: Controller name used in validation error messages.
        entity: Genesis entity used for inverse kinematics.
        dofs_idx: Controlled DoF indices in the entity.
        n_envs: Number of parallel environments.
        device: Torch device used for tensors.
        home_q: Home joint configuration for controlled DoFs, or `None`.
        ik_max_iters: Maximum IK iterations. Defaults to None.
        ik_damping: Damping used by the Genesis IK solver. Defaults to None.
        ik_max_step: Maximum joint step per IK solve. Defaults to None.
        """
        _validate_ik_config(
            controller_name=controller_name,
            ik_max_iters=ik_max_iters,
            ik_damping=ik_damping,
            ik_max_step=ik_max_step,
        )

        self.controller_name = controller_name
        self.entity = entity
        self.dofs_idx = list(dofs_idx)
        self.n_envs = int(n_envs)
        self.device = device
        self.ik_max_iters = None if ik_max_iters is None else int(ik_max_iters)
        self.ik_damping = None if ik_damping is None else float(ik_damping)
        self.ik_max_step = None if ik_max_step is None else float(ik_max_step)
        self._batched = self.n_envs > 1
        self._link = None
        self._home_q_row: torch.Tensor | None = (
            torch.as_tensor(home_q, dtype=torch.float32, device=self.device)
            .contiguous()
            .view(1, -1)
            if home_q is not None
            else None
        )

    @property
    def dof(self) -> int:
        """Return the number of controlled DoFs."""
        return len(self.dofs_idx)

    def build(self, *, end_link_name: str) -> None:
        """Cache the end-effector link used by Genesis IK.

        Args:
            end_link_name: End-effector link name used by the backend.
        """
        self._link = self.entity.get_link(end_link_name)

    def reset(self, env_idx: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset no solver state because Genesis IK is stateless.

        Args:
            env_idx: Environment indices to reset. Defaults to None.
        """
        del env_idx

    def _expect_2d(self, x: torch.Tensor, cols: int) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("Input must be a torch.Tensor.")
        if x.ndim == 1:
            if self.n_envs != 1:
                raise ValueError(
                    f"Expected input shape ({self.n_envs}, {cols}) in batched mode; "
                    "got 1D tensor.",
                )
            if x.numel() != cols:
                raise ValueError(f"Expected input length {cols}, got {x.numel()}.")
            return x.view(1, cols)
        if x.ndim == 2:
            if x.shape[0] != self.n_envs or x.shape[1] != cols:
                raise ValueError(
                    f"Expected input shape ({self.n_envs}, {cols}), "
                    f"got {tuple(x.shape)}.",
                )
            return x
        raise ValueError("Input must be 1D or 2D.")

    def _to_entity_shape(self, x2d: torch.Tensor) -> torch.Tensor:
        return x2d[0] if not self._batched else x2d

    def _full_qpos_template(self, batch: int) -> torch.Tensor:
        qpos = self.entity.get_dofs_position()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[0] != batch:
            qpos = qpos.expand(batch, -1)
        return qpos.to(self.device).contiguous().clone()

    def _q_to_ctrl(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] == self.dof:
            return q
        return q[..., self.dofs_idx]

    def _resolve_full_init_qpos(
        self,
        init_qpos: torch.Tensor | None,
        *,
        batch: int,
    ) -> torch.Tensor:
        full_init_qpos = self._full_qpos_template(batch)
        if init_qpos is None:
            if self._home_q_row is not None:
                full_init_qpos[:, self.dofs_idx] = self._home_q_row.expand(batch, -1)
            return full_init_qpos

        seed = init_qpos if init_qpos.ndim == 2 else init_qpos.view(1, -1)
        seed = self._expand_seed_batch(seed, batch=batch)
        if seed.shape[-1] == full_init_qpos.shape[-1]:
            return seed.contiguous()
        if seed.shape[-1] == len(self.dofs_idx):
            full_init_qpos[:, self.dofs_idx] = seed.contiguous()
            return full_init_qpos
        raise ValueError(
            "init_qpos must match either full entity DoFs or controlled DoFs.",
        )

    @staticmethod
    def _expand_seed_batch(seed: torch.Tensor, *, batch: int) -> torch.Tensor:
        if seed.shape[0] == batch:
            return seed
        if seed.shape[0] == 1:
            return seed.expand(batch, -1)
        raise ValueError(
            f"init_qpos batch dim must be 1 or {batch}; got {seed.shape[0]}.",
        )

    def _ik_kwargs(self) -> dict[str, object]:
        ik_kw: dict[str, object] = {}
        if self.ik_max_iters is not None:
            ik_kw["max_solver_iters"] = self.ik_max_iters
        if self.ik_damping is not None:
            ik_kw["damping"] = self.ik_damping
        if self.ik_max_step is not None:
            ik_kw["max_step_size"] = self.ik_max_step
        return ik_kw

    def _ik(
        self,
        *,
        link: Any,
        pos: torch.Tensor,
        quat: torch.Tensor,
        init_qpos: torch.Tensor | None = None,
        envs_idx: torch.Tensor | None = None,
        restrict_to_control_dofs: bool = False,
        **kw: Any,
    ) -> torch.Tensor:
        """Solve Genesis IK from a full-robot seed and return full-robot qpos."""
        p = self._expect_2d(pos, 3)
        q = self._expect_2d(quat, 4)

        batch = p.shape[0]
        full_init_qpos = self._resolve_full_init_qpos(init_qpos, batch=batch)

        ik_args: dict[str, Any] = {
            "link": link,
            "pos": self._to_entity_shape(p),
            "quat": self._to_entity_shape(q),
            "init_qpos": self._to_entity_shape(full_init_qpos),
        }
        if restrict_to_control_dofs:
            ik_args["dofs_idx_local"] = self.dofs_idx
        if envs_idx is not None:
            ik_args["envs_idx"] = envs_idx

        out = self.entity.inverse_kinematics(**ik_args, **kw)
        return out if self._batched else out.unsqueeze(0)

    def solve_relative_delta(
        self,
        *,
        link_idx_local: int,
        dp: torch.Tensor,
        dr: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a relative end-effector delta using Genesis IK directly.

        Args:
            link_idx_local: Local link index for the end-effector link.
            dp: Linear delta command in meters.
            dr: Angular delta command in radians.
        """
        if self._link is None:
            raise RuntimeError(
                f"{self.controller_name}: backend must be build()'d before use.",
            )

        dp_2d = self._expect_2d(dp, 3).contiguous()
        dr_2d = self._expect_2d(dr, 3).contiguous()
        batch = dp_2d.shape[0]

        qpos = self.entity.get_dofs_position()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[0] != batch:
            qpos = qpos.expand(batch, -1)
        qpos = qpos.to(self.device).contiguous()

        pos_cur, quat_cur = self.entity.forward_kinematics(
            qpos=self._to_entity_shape(qpos),
            links_idx_local=[link_idx_local],
        )
        pos_cur = pos_cur.reshape(-1, 3).to(self.device).contiguous()
        quat_cur = quat_cur.reshape(-1, 4).to(self.device).contiguous()
        pos_tgt = (pos_cur + dp_2d).contiguous()

        quat_delta = rpy_to_quat_wxyz_torch(dr_2d).contiguous()
        quat_tgt = quat_mul_wxyz_torch(quat_delta, quat_cur).contiguous()
        quat_tgt = quat_normalize_torch(quat_tgt).contiguous()

        q = self._ik(
            link=self._link,
            pos=pos_tgt,
            quat=quat_tgt,
            init_qpos=qpos,
            restrict_to_control_dofs=True,
            max_solver_iters=self.ik_max_iters or 80,
            damping=0.01 if self.ik_damping is None else self.ik_damping,
        )
        return self._q_to_ctrl(q).contiguous()

    def solve_targets(
        self,
        *,
        q_init: torch.Tensor | None,
        pos_tgt: torch.Tensor,
        quat_tgt: torch.Tensor,
        iterations: int | None = None,
        integration_dt: float | None = None,
        posture_targets: torch.Tensor | None = None,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve task-space targets into controlled joint positions.

        Args:
            q_init: Initial full-entity joint configuration.
            pos_tgt: Target positions in world-frame meters.
            quat_tgt: Target quaternions in WXYZ order.
            iterations: Maximum solver iterations. Defaults to None.
            integration_dt: Integration time step in seconds. Defaults to None.
            posture_targets: Posture targets used by the backend. Defaults to None.
            hold_mask: Boolean mask for environments that should hold position. Defaults to None.
        """
        del iterations, integration_dt, posture_targets
        if self._link is None:
            raise RuntimeError(
                f"{self.controller_name}: backend must be build()'d before use.",
            )

        q = self._ik(
            link=self._link,
            pos=pos_tgt,
            quat=quat_tgt,
            init_qpos=q_init,
            restrict_to_control_dofs=True,
            **self._ik_kwargs(),
        )
        q_cmd = self._q_to_ctrl(q).contiguous()

        if hold_mask is not None:
            self._apply_hold_mask(q_cmd, hold_mask=hold_mask, q_init=q_init)

        return q_cmd

    def _apply_hold_mask(
        self,
        q_cmd: torch.Tensor,
        *,
        hold_mask: torch.Tensor,
        q_init: torch.Tensor | None,
    ) -> None:
        hold = hold_mask.reshape(-1).to(device=q_cmd.device, dtype=torch.bool)
        if hold.numel() != q_cmd.shape[0]:
            raise ValueError(f"{self.controller_name}: hold_mask batch mismatch.")
        if q_init is None:
            seed = self._full_qpos_template(q_cmd.shape[0])
        else:
            seed = q_init if q_init.ndim == 2 else q_init.view(1, -1)
        if seed.shape[0] == 1 and q_cmd.shape[0] > 1:
            seed = seed.expand(q_cmd.shape[0], -1)
        q_cmd[hold] = self._q_to_ctrl(seed.to(q_cmd.device))[hold]


__all__ = ["GenesisInternalPoseBackend", "make_pose_backend"]
