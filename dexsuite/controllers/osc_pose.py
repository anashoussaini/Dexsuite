"""Provide operational-space pose control with selectable pose backends.

This controller keeps the public ``osc_pose`` action semantics:

    [dx, dy, dz, droll, dpitch, dyaw]

but internally uses Mink to solve a sequence of differential IK substeps toward
an end-effector pose reference. This makes ``osc_pose`` the production task-space
controller while preserving the same external API used by policies and teleop.
"""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.orientation_utils import (
    quat_mul_wxyz_torch,
    quat_normalize_torch,
    rpy_to_quat_wxyz_torch,
)
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller
from .pose_backend import make_pose_backend
from .validation import (
    normalize_pose_backend,
    require_bool,
    require_limit_gain,
    require_nonnegative_float,
    require_optional_positive_float,
    require_optional_positive_int,
    require_positive_float,
    require_positive_int,
)


@register_controller("osc_pose")
class OSCPoseController(Controller):
    """Control relative end-effector pose deltas using differential IK."""

    def __init__(
        self,
        *,
        entity: Any,
        model: Any,
        normalized: bool | None,
        clip: bool | None,
        lin_scale: float | None,
        ang_scale: float | None,
        mink_dt: float | None = 0.05,
        mink_substeps: int | None = 8,
        solver: str | None = "quadprog",
        frame_type: str | None = "body",
        lin_speed_limit: float | None = 0.20,
        ang_speed_limit: float | None = 1.00,
        lin_accel_limit: float | None = 4.0,
        ang_accel_limit: float | None = 8.0,
        hard_stop_on_zero: bool = True,
        backend: str = "mink",
        ik_max_iters: int | None = None,
        ik_damping: float | None = None,
        position_cost: float | list[float] | tuple[float, ...] = 1.0,
        orientation_cost: float | list[float] | tuple[float, ...] = 5.0,
        posture_cost: float | list[float] | tuple[float, ...] = 1e-3,
        damping_cost: float | list[float] | tuple[float, ...] = 1e-3,
        lm_damping: float = 1.0,
        qp_damping: float = 1e-6,
        max_joint_step: float | None = 0.05,
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
        clip: Whether to clip actions before solving.
        lin_scale: Linear action scale in meters.
        ang_scale: Angular action scale in radians.
        mink_dt: Mink control time step in seconds. Defaults to 0.05.
        mink_substeps: Number of differential IK substeps. Defaults to 8.
        solver: QP solver name. Defaults to "quadprog".
        frame_type: Backend task frame type. Defaults to "body".
        lin_speed_limit: Linear speed limit in meters per second. Defaults to 0.20.
        ang_speed_limit: Angular speed limit in radians per second. Defaults to 1.00.
        lin_accel_limit: Linear acceleration limit. Defaults to 4.0.
        ang_accel_limit: Angular acceleration limit. Defaults to 8.0.
        hard_stop_on_zero: Whether zero actions reset velocity references. Defaults to True.
        backend: Pose backend name or alias. Defaults to "mink".
        ik_max_iters: Maximum IK iterations for backends that need it. Defaults to None.
        ik_damping: IK damping for backends that need it. Defaults to None.
        position_cost: Backend position task cost. Defaults to 1.0.
        orientation_cost: Backend orientation task cost. Defaults to 5.0.
        posture_cost: Backend posture task cost. Defaults to 1e-3.
        damping_cost: Backend damping task cost. Defaults to 1e-3.
        lm_damping: Levenberg-Marquardt damping value. Defaults to 1.0.
        qp_damping: QP damping value. Defaults to 1e-6.
        max_joint_step: Maximum joint step per solve. Defaults to 0.05.
        configuration_limit_gain: Configuration-limit gain in `(0, 1]`. Defaults to 0.95.
        pos_tol: Position tolerance for convergence. Defaults to 1e-4.
        rot_tol: Rotation tolerance for convergence. Defaults to 1e-4.
        kw: Additional keyword arguments forwarded to `Controller`.
        """
        super().__init__(entity=entity, model=model, **kw)

        controller_name = "osc_pose"
        if normalized is None or clip is None or lin_scale is None or ang_scale is None:
            raise ValueError(
                "osc_pose: 'normalized', 'clip', 'lin_scale', and "
                "'ang_scale' are required.",
            )

        lin_scale_value = require_positive_float(
            controller_name, "lin_scale", lin_scale
        )
        ang_scale_value = require_positive_float(
            controller_name, "ang_scale", ang_scale
        )
        mink_dt_value = require_positive_float(controller_name, "mink_dt", mink_dt)
        mink_substeps_value = require_positive_int(
            controller_name,
            "mink_substeps",
            mink_substeps,
        )
        lin_speed_limit_value = require_optional_positive_float(
            controller_name,
            "lin_speed_limit",
            lin_speed_limit,
        )
        ang_speed_limit_value = require_optional_positive_float(
            controller_name,
            "ang_speed_limit",
            ang_speed_limit,
        )
        lin_accel_limit_value = require_optional_positive_float(
            controller_name,
            "lin_accel_limit",
            lin_accel_limit,
        )
        ang_accel_limit_value = require_optional_positive_float(
            controller_name,
            "ang_accel_limit",
            ang_accel_limit,
        )
        lm_damping_value = require_nonnegative_float(
            controller_name,
            "lm_damping",
            lm_damping,
        )
        qp_damping_value = require_nonnegative_float(
            controller_name,
            "qp_damping",
            qp_damping,
        )
        max_joint_step_value = require_optional_positive_float(
            controller_name,
            "max_joint_step",
            max_joint_step,
        )
        configuration_limit_gain_value = require_limit_gain(
            controller_name,
            configuration_limit_gain,
        )
        pos_tol_value = require_nonnegative_float(controller_name, "pos_tol", pos_tol)
        rot_tol_value = require_nonnegative_float(controller_name, "rot_tol", rot_tol)
        ik_max_iters_value = require_optional_positive_int(
            controller_name,
            "ik_max_iters",
            ik_max_iters,
        )
        ik_damping_value = (
            None
            if ik_damping is None
            else require_nonnegative_float(controller_name, "ik_damping", ik_damping)
        )

        self.normalized = require_bool(controller_name, "normalized", normalized)
        self.clip = require_bool(controller_name, "clip", clip)
        self.backend = normalize_pose_backend(controller_name, backend)
        self.lin_scale = lin_scale_value
        self.ang_scale = ang_scale_value
        self.mink_dt = mink_dt_value
        self.mink_substeps = mink_substeps_value
        self._integration_dt = self.mink_dt / float(self.mink_substeps)
        self.lin_speed_limit = lin_speed_limit_value
        self.ang_speed_limit = ang_speed_limit_value
        self.lin_accel_limit = lin_accel_limit_value
        self.ang_accel_limit = ang_accel_limit_value
        self.hard_stop_on_zero = require_bool(
            controller_name,
            "hard_stop_on_zero",
            hard_stop_on_zero,
        )
        self.ik_max_iters = ik_max_iters_value
        self.ik_damping = ik_damping_value

        self._link = self._get_end_effector_link()
        self._link_idx = self._link.idx_local
        self._backend = make_pose_backend(
            controller_name=controller_name,
            backend=self.backend,
            entity=self.entity,
            model=self.model,
            dofs_idx=self.dofs_idx,
            n_envs=self.n_envs,
            device=self.device,
            home_q=self.home_q,
            dt=self.mink_dt,
            solver=solver,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            posture_cost=posture_cost,
            damping_cost=damping_cost,
            lm_damping=lm_damping_value,
            qp_damping=qp_damping_value,
            max_joint_step=max_joint_step_value,
            configuration_limit_gain=configuration_limit_gain_value,
            pos_tol=pos_tol_value,
            rot_tol=rot_tol_value,
            ik_max_iters=self.ik_max_iters,
            ik_damping=self.ik_damping,
        )

        self._twist_ref: torch.Tensor | None = None
        self._pos_ref: torch.Tensor | None = None
        self._quat_ref: torch.Tensor | None = None
        self._ref_valid: torch.Tensor | None = None

    def post_build(self) -> None:
        """Build backend state after entity creation."""
        super().post_build()
        self._backend.build(end_link_name=self.model.end_link)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Clear controller and backend solver state on environment reset.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        self._backend.reset(env_idx=env_idx)
        indices = self._env_indices(env_idx)
        if indices is None:
            self._twist_ref = None
            self._pos_ref = None
            self._quat_ref = None
            self._ref_valid = None
            return
        if self._ref_valid is not None:
            self._ref_valid[indices] = False
        if self._twist_ref is not None:
            self._twist_ref[indices] = 0.0

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
            [
                -self.lin_scale,
                -self.lin_scale,
                -self.lin_scale,
                -self.ang_scale,
                -self.ang_scale,
                -self.ang_scale,
            ],
            dtype=torch.float32,
        )
        return float_box(low, -low)

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        if self.backend == "genesis_internal":
            self._apply_cmd_genesis_internal(vec_2d)
            return

        v = self._expect_2d(vec_2d, 6).contiguous()
        dp = v[:, :3].contiguous()
        dr = v[:, 3:].contiguous()

        if self.normalized:
            if self.clip:
                dp = dp.clamp(-1.0, 1.0)
                dr = dr.clamp(-1.0, 1.0)
            dp = dp * self.lin_scale
            dr = dr * self.ang_scale
        elif self.clip:
            dp = dp.clamp(-self.lin_scale, self.lin_scale)
            dr = dr.clamp(-self.ang_scale, self.ang_scale)

        desired_twist = torch.cat(
            [dp / self.mink_dt, dr / self.mink_dt],
            dim=1,
        ).contiguous()
        twist_cmd = self._shape_twist(desired_twist).contiguous()
        dp = (twist_cmd[:, :3] * self.mink_dt).contiguous()
        dr = (twist_cmd[:, 3:] * self.mink_dt).contiguous()

        q_cur = self._full_qpos_template(v.shape[0]).contiguous()
        pos_cur, quat_cur = self._fk_pos_quat_2d(self._link_idx)
        self._ensure_reference_state(pos_cur, quat_cur)

        zero_mask = (dp.abs().amax(dim=1) <= 1e-8) & (dr.abs().amax(dim=1) <= 1e-8)
        quat_delta = rpy_to_quat_wxyz_torch(dr).contiguous()

        if zero_mask.any():
            self._pos_ref[zero_mask] = pos_cur[zero_mask]
            self._quat_ref[zero_mask] = quat_cur[zero_mask]

        move_mask = ~zero_mask
        if move_mask.any():
            self._pos_ref[move_mask] = self._pos_ref[move_mask] + dp[move_mask]
            self._quat_ref[move_mask] = quat_mul_wxyz_torch(
                quat_delta[move_mask],
                self._quat_ref[move_mask],
            )
            self._quat_ref[move_mask] = quat_normalize_torch(self._quat_ref[move_mask])

        q_cmd = self._backend.solve_targets(
            q_init=q_cur,
            pos_tgt=self._pos_ref.contiguous(),
            quat_tgt=quat_normalize_torch(self._quat_ref).contiguous(),
            iterations=self.mink_substeps,
            integration_dt=self._integration_dt,
            hold_mask=zero_mask if self.hard_stop_on_zero else None,
        )
        self._control_position(q_cmd)

    def _ensure_reference_state(
        self,
        pos_cur: torch.Tensor,
        quat_cur: torch.Tensor,
    ) -> None:
        """Initialize or refresh invalid per-env pose references."""
        if (
            self._pos_ref is None
            or self._quat_ref is None
            or self._pos_ref.shape != pos_cur.shape
            or self._quat_ref.shape != quat_cur.shape
        ):
            self._pos_ref = pos_cur.clone()
            self._quat_ref = quat_cur.clone()
            self._ref_valid = torch.ones(
                pos_cur.shape[0],
                dtype=torch.bool,
                device=pos_cur.device,
            )
            return

        if self._ref_valid is None or self._ref_valid.shape[0] != pos_cur.shape[0]:
            self._ref_valid = torch.ones(
                pos_cur.shape[0],
                dtype=torch.bool,
                device=pos_cur.device,
            )
            return

        invalid = ~self._ref_valid
        if invalid.any():
            self._pos_ref[invalid] = pos_cur[invalid]
            self._quat_ref[invalid] = quat_cur[invalid]
            if self._twist_ref is not None:
                self._twist_ref[invalid] = 0.0
            self._ref_valid[invalid] = True

    def _apply_cmd_genesis_internal(self, vec_2d: torch.Tensor) -> None:
        """Apply the Genesis internal relative OSC command path."""
        v = self._expect_2d(vec_2d, 6).contiguous()
        dp = v[:, :3].contiguous()
        dr = v[:, 3:].contiguous()

        if self.normalized:
            if self.clip:
                dp = dp.clamp(-1.0, 1.0)
                dr = dr.clamp(-1.0, 1.0)
            dp = dp * self.lin_scale
            dr = dr * self.ang_scale
        elif self.clip:
            dp = dp.clamp(-self.lin_scale, self.lin_scale)
            dr = dr.clamp(-self.ang_scale, self.ang_scale)

        q_cmd = self._backend.solve_relative_delta(
            link_idx_local=self._link_idx,
            dp=dp,
            dr=dr,
        )
        self._control_position(q_cmd)

    def _shape_twist(self, desired_twist: torch.Tensor) -> torch.Tensor:
        desired = self._clamp_twist_limits(desired_twist.clone())

        if self._twist_ref is None or self._twist_ref.shape != desired.shape:
            self._twist_ref = torch.zeros_like(desired)

        twist = self._twist_ref
        zero_mask = desired.abs().amax(dim=1) <= 1e-8
        move_mask = ~zero_mask

        if zero_mask.any() and self.hard_stop_on_zero:
            twist[zero_mask] = 0.0

        if move_mask.any():
            twist[move_mask] = twist[move_mask] + self._limit_twist_delta(
                desired[move_mask] - twist[move_mask],
            )

        if zero_mask.any() and not self.hard_stop_on_zero:
            twist[zero_mask] = self._decay_twist(twist[zero_mask])

        return twist.clone()

    def _clamp_twist_limits(self, twist: torch.Tensor) -> torch.Tensor:
        """Clamp linear and angular twist magnitudes to configured speed limits."""
        if self.lin_speed_limit is not None:
            twist[:, :3] = twist[:, :3].clamp(
                -self.lin_speed_limit,
                self.lin_speed_limit,
            )
        if self.ang_speed_limit is not None:
            twist[:, 3:] = twist[:, 3:].clamp(
                -self.ang_speed_limit,
                self.ang_speed_limit,
            )
        return twist

    def _limit_twist_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Clamp twist deltas to configured acceleration limits."""
        if self.lin_accel_limit is not None:
            lin_step = self.lin_accel_limit * self.mink_dt
            delta[:, :3] = delta[:, :3].clamp(-lin_step, lin_step)
        if self.ang_accel_limit is not None:
            ang_step = self.ang_accel_limit * self.mink_dt
            delta[:, 3:] = delta[:, 3:].clamp(-ang_step, ang_step)
        return delta

    def _decay_twist(self, twist: torch.Tensor) -> torch.Tensor:
        """Relax twist toward zero while respecting acceleration limits."""
        if self.lin_accel_limit is None and self.ang_accel_limit is None:
            return torch.zeros_like(twist)
        return twist + self._limit_twist_delta(-twist)
