"""Provide quaternion absolute-offset OSC with selectable pose backends."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.orientation_utils import (
    quat_mul_wxyz_torch,
    quat_normalize_torch,
)
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller
from .pose_backend import make_pose_backend
from .validation import (
    normalize_pose_backend,
    require_bool,
    require_limit_gain,
    require_nonnegative_float,
    require_positive_float,
    require_positive_int,
    require_positive_vector,
)


@register_controller("osc_pose_abs_quat")
class OSCPoseAbsoluteQuatController(Controller):
    """Control absolute end-effector offsets with quaternion rotations."""

    def __init__(
        self,
        *,
        entity: Any,
        model: Any,
        normalized: bool | None,
        clip: bool | None,
        lin_range: tuple[float, float, float] | None,
        ang_range: tuple[float, float, float] | None = None,
        backend: str = "mink",
        mink_dt: float = 0.05,
        mink_substeps: int = 8,
        solver: str = "quadprog",
        frame_type: str = "body",
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
        ik_max_iters: int = 100,
        ik_damping: float = 0.05,
        **kw: Any,
    ) -> None:
        """Initialize the object.

        Args:
        entity: Genesis entity to control.
        model: Robot component model associated with the entity.
        normalized: Whether actions are normalized.
        clip: Whether to clip actions before solving.
        lin_range: Linear offset range in meters.
        ang_range: Angular offset range in radians.
        backend: Pose backend name or alias. Defaults to "mink".
        mink_dt: Mink control time step in seconds. Defaults to 0.05.
        mink_substeps: Number of differential IK substeps. Defaults to 8.
        solver: QP solver name. Defaults to "quadprog".
        frame_type: Backend task frame type. Defaults to "body".
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
        ik_max_iters: Maximum IK iterations for backends that need it. Defaults to None.
        ik_damping: IK damping for backends that need it. Defaults to None.
        kw: Additional keyword arguments forwarded to `Controller`.
        """
        super().__init__(entity=entity, model=model, **kw)

        controller_name = "osc_pose_abs_quat"
        del ang_range
        if normalized is None or clip is None or lin_range is None:
            raise ValueError(
                "osc_pose_abs_quat: 'normalized', 'clip', and 'lin_range' "
                "are required.",
            )

        lin_range_value = require_positive_vector(
            controller_name,
            "lin_range",
            lin_range,
            length=3,
        )
        mink_dt_value = require_positive_float(controller_name, "mink_dt", mink_dt)
        mink_substeps_value = require_positive_int(
            controller_name,
            "mink_substeps",
            mink_substeps,
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
        max_joint_step_value = (
            None
            if max_joint_step is None
            else require_positive_float(
                controller_name, "max_joint_step", max_joint_step
            )
        )
        configuration_limit_gain_value = require_limit_gain(
            controller_name,
            configuration_limit_gain,
        )
        pos_tol_value = require_nonnegative_float(controller_name, "pos_tol", pos_tol)
        rot_tol_value = require_nonnegative_float(controller_name, "rot_tol", rot_tol)

        self.normalized = require_bool(controller_name, "normalized", normalized)
        self.clip = require_bool(controller_name, "clip", clip)
        self.backend = normalize_pose_backend(controller_name, backend)
        self.lin_range = torch.tensor(
            lin_range_value,
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self.mink_dt = mink_dt_value
        self.mink_substeps = mink_substeps_value
        self._integration_dt = self.mink_dt / float(self.mink_substeps)
        self.ik_max_iters = ik_max_iters_value
        self.ik_damping = ik_damping_value

        self._link = self._get_end_effector_link()
        self._link_idx = self._link.idx_local
        self._origin_pos: torch.Tensor | None = None
        self._origin_quat: torch.Tensor | None = None
        self._origin_valid: torch.Tensor | None = None
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

    def post_build(self) -> None:
        """Build backend state after entity creation."""
        super().post_build()
        self._backend.build(end_link_name=self.model.end_link)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Clear absolute-origin and backend state on environment reset.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        self._backend.reset(env_idx=env_idx)
        indices = self._env_indices(env_idx)
        if indices is None:
            self._origin_pos = None
            self._origin_quat = None
            self._origin_valid = None
            return
        if self._origin_valid is not None:
            self._origin_valid[indices] = False

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
            return float_box(-1.0, 1.0, shape=(7,))
        quat_bounds = torch.ones(4, dtype=torch.float32, device=self.device)
        low = torch.cat((-self.lin_range, -quat_bounds), dim=0)
        return float_box(low, -low)

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        v = self._expect_2d(vec_2d, 7).contiguous()
        self._ensure_origin_state()
        dp = v[:, :3].contiguous()
        dq = v[:, 3:7].contiguous()

        if self.normalized:
            if self.clip:
                dp = dp.clamp(-1.0, 1.0)
                dq = dq.clamp(-1.0, 1.0)
            dp = dp * self.lin_range
        elif self.clip:
            dp = torch.min(torch.max(dp, -self.lin_range), self.lin_range)
            dq = dq.clamp(-1.0, 1.0)

        dq = quat_normalize_torch(dq).contiguous()
        pos_tgt = (self._origin_pos + dp).contiguous()
        quat_tgt = quat_mul_wxyz_torch(dq, self._origin_quat).contiguous()
        quat_tgt = quat_normalize_torch(quat_tgt).contiguous()

        q_seed = self._full_qpos_template(v.shape[0]).contiguous()
        q_cmd = self._backend.solve_targets(
            q_init=q_seed,
            pos_tgt=pos_tgt,
            quat_tgt=quat_tgt,
            iterations=self.mink_substeps,
            integration_dt=self._integration_dt,
        )
        self._control_position(q_cmd)

    def _ensure_origin_state(self) -> None:
        """Initialize or refresh invalid per-env absolute origins."""
        pos_cur, quat_cur = self._fk_pos_quat_2d(self._link_idx)
        if (
            self._origin_pos is None
            or self._origin_quat is None
            or self._origin_pos.shape != pos_cur.shape
            or self._origin_quat.shape != quat_cur.shape
        ):
            self._origin_pos = pos_cur.clone()
            self._origin_quat = quat_cur.clone()
            self._origin_valid = torch.ones(
                pos_cur.shape[0],
                dtype=torch.bool,
                device=pos_cur.device,
            )
            return

        if (
            self._origin_valid is None
            or self._origin_valid.shape[0] != pos_cur.shape[0]
        ):
            self._origin_valid = torch.ones(
                pos_cur.shape[0],
                dtype=torch.bool,
                device=pos_cur.device,
            )
            return

        invalid = ~self._origin_valid
        if invalid.any():
            self._origin_pos[invalid] = pos_cur[invalid]
            self._origin_quat[invalid] = quat_cur[invalid]
            self._origin_valid[invalid] = True
