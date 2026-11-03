"""Provide absolute end-effector IK control with selectable pose backends."""

from __future__ import annotations

import math
from typing import Any

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.orientation_utils import rpy_to_quat_wxyz_torch
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller
from .pose_backend import make_pose_backend
from .validation import (
    normalize_pose_backend,
    require_bool,
    require_choice,
    require_limit_gain,
    require_nonnegative_float,
    require_positive_float,
    require_positive_int,
)


@register_controller("ik_pose")
class IKPoseController(Controller):
    """Control absolute end-effector pose targets using iterative IK."""

    def __init__(
        self,
        *,
        entity: Any,
        model: Any,
        normalized: bool | None = True,
        ik_max_iters: int | None,
        ik_damping: float | None,
        ik_max_step: float | None,
        ik_init: str | None,
        clip: bool | None = None,
        backend: str = "mink",
        ik_dt: float = 0.05,
        solver: str = "quadprog",
        frame_type: str = "body",
        position_cost: float | list[float] | tuple[float, ...] = 1.0,
        orientation_cost: float | list[float] | tuple[float, ...] = 5.0,
        posture_cost: float | list[float] | tuple[float, ...] = 1e-3,
        damping_cost: float | list[float] | tuple[float, ...] = 1e-3,
        lm_damping: float | None = None,
        qp_damping: float = 1e-6,
        configuration_limit_gain: float = 0.95,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-4,
        **kw: Any,
    ) -> None:
        """Initialize the object.

        Args:
        entity: Genesis entity to control.
        model: Robot component model associated with the entity.
        normalized: Whether actions are normalized.
        ik_max_iters: Maximum IK iterations.
        ik_damping: IK damping value.
        ik_max_step: Maximum joint step per IK solve.
        ik_init: Seed policy for IK initialization.
        clip: Whether to clip actions before solving. Defaults to None.
        backend: Pose backend name or alias. Defaults to "mink".
        ik_dt: IK integration time step in seconds. Defaults to 0.05.
        solver: QP solver name. Defaults to "quadprog".
        frame_type: Backend task frame type. Defaults to "body".
        position_cost: Backend position task cost. Defaults to 1.0.
        orientation_cost: Backend orientation task cost. Defaults to 5.0.
        posture_cost: Backend posture task cost. Defaults to 1e-3.
        damping_cost: Backend damping task cost. Defaults to 1e-3.
        lm_damping: Levenberg-Marquardt damping override. Defaults to None.
        qp_damping: QP damping value. Defaults to 1e-6.
        configuration_limit_gain: Configuration-limit gain in `(0, 1]`. Defaults to 0.95.
        pos_tol: Position tolerance for convergence. Defaults to 1e-4.
        rot_tol: Rotation tolerance for convergence. Defaults to 1e-4.
        kw: Additional keyword arguments forwarded to `Controller`.
        """
        super().__init__(entity=entity, model=model, **kw)

        controller_name = "ik_pose"
        if normalized is None:
            raise ValueError("ik_pose: 'normalized' is required.")
        if ik_max_iters is None or ik_damping is None or ik_init is None:
            raise ValueError(
                "ik_pose: 'ik_max_iters', 'ik_damping', and 'ik_init' are required.",
            )

        ik_max_iters_value = require_positive_int(
            controller_name,
            "ik_max_iters",
            ik_max_iters,
        )
        ik_damping_value = require_nonnegative_float(
            controller_name,
            "ik_damping",
            ik_damping,
        )
        ik_max_step_value = (
            None
            if ik_max_step is None
            else require_positive_float(controller_name, "ik_max_step", ik_max_step)
        )
        ik_dt_value = require_positive_float(controller_name, "ik_dt", ik_dt)
        lm_damping_value = (
            ik_damping_value
            if lm_damping is None
            else require_nonnegative_float(controller_name, "lm_damping", lm_damping)
        )
        qp_damping_value = require_nonnegative_float(
            controller_name,
            "qp_damping",
            qp_damping,
        )
        configuration_limit_gain_value = require_limit_gain(
            controller_name,
            configuration_limit_gain,
        )
        pos_tol_value = require_nonnegative_float(controller_name, "pos_tol", pos_tol)
        rot_tol_value = require_nonnegative_float(controller_name, "rot_tol", rot_tol)

        self.normalized = require_bool(controller_name, "normalized", normalized)
        self.ik_max_iters = ik_max_iters_value
        self.ik_damping = ik_damping_value
        self.ik_max_step = ik_max_step_value
        self.ik_init = require_choice(
            controller_name,
            "ik_init",
            str(ik_init),
            {"home", "current"},
        )
        self.clip = (
            False if clip is None else require_bool(controller_name, "clip", clip)
        )
        self.backend = normalize_pose_backend(controller_name, backend)
        self.ik_dt = ik_dt_value
        self._link = self._get_end_effector_link()

        self._backend = make_pose_backend(
            controller_name=controller_name,
            backend=self.backend,
            entity=self.entity,
            model=self.model,
            dofs_idx=self.dofs_idx,
            n_envs=self.n_envs,
            device=self.device,
            home_q=self.home_q,
            dt=self.ik_dt,
            solver=solver,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            posture_cost=posture_cost,
            damping_cost=damping_cost,
            lm_damping=lm_damping_value,
            qp_damping=qp_damping_value,
            max_joint_step=self.ik_max_step,
            configuration_limit_gain=configuration_limit_gain_value,
            pos_tol=pos_tol_value,
            rot_tol=rot_tol_value,
            ik_max_iters=self.ik_max_iters,
            ik_damping=self.ik_damping,
            ik_max_step=self.ik_max_step,
        )

    def post_build(self) -> None:
        """Build backend state after entity creation."""
        super().post_build()
        self._backend.build(end_link_name=self.model.end_link)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Clear backend solver state on environment reset.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        self._backend.reset(env_idx=env_idx)

    def set_home_position(self, env_idx: torch.Tensor | None = None) -> None:
        """Reset home state using state-only Genesis semantics when selected.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        if self.backend == "genesis_internal":
            self._set_home_position_state_only(env_idx=env_idx)
            return
        super().set_home_position(env_idx=env_idx)

    def action_space(self) -> spaces.Box:
        """Return the controller action space."""
        if self.normalized:
            return float_box(-1.0, 1.0, shape=(6,))
        low = torch.tensor(
            [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
            dtype=torch.float32,
        )
        return float_box(low, -low)

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        v = self._expect_2d(vec_2d, 6).contiguous()
        if self.clip and self.normalized:
            v = v.clamp(-1.0, 1.0)

        pos = v[:, :3].contiguous()
        rpy = v[:, 3:].contiguous()

        if self.normalized:
            rpy = rpy * float(math.pi)
        elif self.clip:
            pos = pos.clamp(-1.0, 1.0)
            rpy = rpy.clamp(-math.pi, math.pi)

        quat = rpy_to_quat_wxyz_torch(rpy).contiguous()

        q_current = self._full_qpos_template(v.shape[0]).contiguous()
        if self.ik_init == "home":
            q_seed = q_current.clone()
            if self.home_q is None or self.home_q.numel() != self.dof:
                q_seed[:, self.dofs_idx] = 0.0
            else:
                q_seed[:, self.dofs_idx] = self.home_q.to(self.device).expand(
                    v.shape[0],
                    -1,
                )
        else:
            q_seed = q_current

        q_cmd = self._backend.solve_targets(
            q_init=q_seed,
            pos_tgt=pos,
            quat_tgt=quat,
            iterations=self.ik_max_iters,
            integration_dt=self.ik_dt,
            posture_targets=q_seed,
        )
        self._control_position(q_cmd)
