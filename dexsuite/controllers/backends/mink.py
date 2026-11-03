"""Provide the Mink-backed differential IK backend for pose controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mink
import mujoco
import numpy as np
import qpsolvers
import torch

from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch

if TYPE_CHECKING:
    from collections.abc import Sequence

CostSpec = float | list[float] | tuple[float, ...]


def _resolve_model_path(*, controller_name: str, model: Any) -> Path:
    model_path = getattr(model, "model_path", None)
    if not model_path:
        raise ValueError(f"{controller_name}: model must define model_path.")

    resolved = Path(model_path).expanduser()
    if not resolved.exists():
        raise ValueError(f"{controller_name}: model_path does not exist: {resolved}")
    if resolved.suffix.lower() != ".xml":
        raise ValueError(
            f"{controller_name}: Mink currently requires an MJCF/XML model_path.",
        )
    return resolved


def _validate_backend_options(
    *,
    controller_name: str,
    solver: str,
    frame_type: str,
) -> None:
    if frame_type not in mink.SUPPORTED_FRAMES:
        raise ValueError(
            f"{controller_name}: frame_type must be one of "
            f"{sorted(mink.SUPPORTED_FRAMES)}, got {frame_type!r}.",
        )

    available = set(qpsolvers.available_solvers)
    if solver not in available:
        raise ValueError(
            f"{controller_name}: solver {solver!r} is not available. "
            f"Available solvers: {sorted(available)}.",
        )


def as_cost_array(
    value: CostSpec,
    *,
    controller_name: str,
    name: str,
) -> np.ndarray:
    """Validate and normalize a Mink task cost specification.

    Args:
        value: Candidate value to validate or normalize.
        controller_name: Controller name used in validation error messages.
        name: Option name used in validation error messages.
    """
    cost = np.asarray(value, dtype=np.float64)
    if cost.ndim > 1:
        raise ValueError(f"{controller_name}: {name} must be a scalar or 1D sequence.")
    if cost.size == 0:
        raise ValueError(f"{controller_name}: {name} cannot be empty.")
    if not np.all(np.isfinite(cost)):
        raise ValueError(f"{controller_name}: {name} must be finite.")
    if np.any(cost < 0.0):
        raise ValueError(f"{controller_name}: {name} must be >= 0.")
    return cost


def make_pose_backend(
    *,
    controller_name: str,
    model: Any,
    dofs_idx: list[int],
    n_envs: int,
    device: torch.device,
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
) -> MinkPoseBackend:
    """Build a ``MinkPoseBackend`` from controller-facing config values.

    Args:
        controller_name: Controller name used in validation error messages.
        model: Robot component model associated with the entity.
        dofs_idx: Controlled DoF indices in the entity.
        n_envs: Number of parallel environments.
        device: Torch device used for tensors.
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
    """
    return MinkPoseBackend(
        controller_name=controller_name,
        model=model,
        dofs_idx=dofs_idx,
        n_envs=n_envs,
        device=device,
        control_dt=dt,
        velocity_dt=dt,
        solver=str(solver or "quadprog"),
        frame_type=str(frame_type or "body"),
        position_cost=as_cost_array(
            position_cost,
            controller_name=controller_name,
            name="position_cost",
        ),
        orientation_cost=as_cost_array(
            orientation_cost,
            controller_name=controller_name,
            name="orientation_cost",
        ),
        posture_cost=as_cost_array(
            posture_cost,
            controller_name=controller_name,
            name="posture_cost",
        ),
        damping_cost=as_cost_array(
            damping_cost,
            controller_name=controller_name,
            name="damping_cost",
        ),
        lm_damping=lm_damping,
        qp_damping=qp_damping,
        max_joint_step=max_joint_step,
        configuration_limit_gain=configuration_limit_gain,
        pos_tol=pos_tol,
        rot_tol=rot_tol,
    )


@dataclass(slots=True)
class _MinkEnvState:
    configuration: mink.Configuration
    position_task: mink.FrameTask
    orientation_task: mink.FrameTask
    posture_task: mink.PostureTask
    damping_task: mink.DampingTask
    limits: tuple[mink.Limit, ...]
    constraints: tuple[mink.Task, ...]
    posture_target_q: np.ndarray | None = None

    @property
    def tasks(
        self,
    ) -> tuple[mink.FrameTask, mink.FrameTask, mink.PostureTask, mink.DampingTask]:
        """Return the task sequence passed to Mink in solver order."""
        return (
            self.position_task,
            self.orientation_task,
            self.posture_task,
            self.damping_task,
        )


class MinkPoseBackend:
    """Manage shared per-env Mink state and solve loops for pose controllers."""

    def __init__(
        self,
        *,
        controller_name: str,
        model: Any,
        dofs_idx: list[int],
        n_envs: int,
        device: torch.device,
        control_dt: float,
        velocity_dt: float | None = None,
        solver: str = "quadprog",
        frame_type: str = "body",
        position_cost: np.ndarray,
        orientation_cost: np.ndarray,
        posture_cost: np.ndarray,
        damping_cost: np.ndarray,
        lm_damping: float,
        qp_damping: float,
        max_joint_step: float | None,
        configuration_limit_gain: float,
        pos_tol: float,
        rot_tol: float,
    ) -> None:
        """Initialize the object.

        Args:
        controller_name: Controller name used in validation error messages.
        model: Robot component model associated with the controlled entity.
        dofs_idx: Controlled DoF indices in the entity.
        n_envs: Number of parallel environments.
        device: Torch device used for tensors.
        control_dt: Controller time step in seconds.
        velocity_dt: Velocity integration time step in seconds.
        solver: Mink QP solver name.
        frame_type: Task frame type passed to Mink.
        position_cost: Position task cost specification.
        orientation_cost: Orientation task cost specification.
        posture_cost: Posture task cost specification.
        damping_cost: Damping task cost specification.
        lm_damping: Levenberg-Marquardt damping value.
        qp_damping: QP damping value.
        max_joint_step: Maximum joint step per solve, or `None`.
        configuration_limit_gain: Configuration-limit gain in `(0, 1]`.
        pos_tol: Position tolerance for convergence.
        rot_tol: Rotation tolerance for convergence.
        """
        self.controller_name = controller_name
        self.model = model
        self.dofs_idx = list(dofs_idx)
        self.n_envs = int(n_envs)
        self.device = device
        self.control_dt = float(control_dt)
        self.velocity_dt = float(velocity_dt if velocity_dt is not None else control_dt)
        self.solver = str(solver).lower()
        self.frame_type = str(frame_type).lower()
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.posture_cost = posture_cost
        self.damping_cost = damping_cost
        self.lm_damping = float(lm_damping)
        self.qp_damping = float(qp_damping)
        self.max_joint_step = None if max_joint_step is None else float(max_joint_step)
        self.configuration_limit_gain = float(configuration_limit_gain)
        self.pos_tol = float(pos_tol)
        self.rot_tol = float(rot_tol)

        self._model_path = _resolve_model_path(
            controller_name=controller_name,
            model=model,
        )
        _validate_backend_options(
            controller_name=controller_name,
            solver=self.solver,
            frame_type=self.frame_type,
        )

        self._end_link_name: str | None = None
        self._mujoco_model: mujoco.MjModel | None = None
        self._states: list[_MinkEnvState] = []

    def build(self, *, end_link_name: str) -> None:
        """Create the shared MuJoCo model and per-env Mink states.

        Args:
            end_link_name: End-effector link name used by the backend.
        """
        self._end_link_name = str(end_link_name)
        self._mujoco_model = mujoco.MjModel.from_xml_path(self._model_path.as_posix())
        self._states = [
            self._make_state(self._mujoco_model, self._end_link_name)
            for _ in range(self.n_envs)
        ]

    def reset(self, env_idx: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset all or selected per-env Mink solver states.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        if self._mujoco_model is None or self._end_link_name is None:
            self._states.clear()
            return

        if env_idx is None or len(self._states) != self.n_envs:
            self._states = [
                self._make_state(self._mujoco_model, self._end_link_name)
                for _ in range(self.n_envs)
            ]
            return

        for index in self._env_indices(env_idx):
            self._states[index] = self._make_state(
                self._mujoco_model,
                self._end_link_name,
            )

    def _env_indices(self, env_idx: torch.Tensor | Sequence[int]) -> list[int]:
        indices = torch.as_tensor(env_idx, dtype=torch.long, device="cpu").view(-1)
        if not ((indices >= 0) & (indices < self.n_envs)).all().item():
            raise ValueError(
                f"{self.controller_name}: env_idx {indices.tolist()} out of range "
                f"[0, {self.n_envs}).",
            )
        return [int(index) for index in indices.tolist()]

    def _normalize_solve_inputs(
        self,
        *,
        q_init: torch.Tensor,
        pos_tgt: torch.Tensor,
        quat_tgt: torch.Tensor,
        posture_targets: torch.Tensor | None,
        hold_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Normalize solve inputs to consistent 2D torch tensors."""
        q_init_2d = q_init if q_init.ndim == 2 else q_init.view(1, -1)
        pos_tgt_2d = pos_tgt if pos_tgt.ndim == 2 else pos_tgt.view(1, -1)
        quat_tgt_2d = quat_tgt if quat_tgt.ndim == 2 else quat_tgt.view(1, -1)

        batch_size = pos_tgt_2d.shape[0]
        if q_init_2d.shape[0] != batch_size or quat_tgt_2d.shape[0] != batch_size:
            raise ValueError(
                f"{self.controller_name}: batch mismatch between q_init/pos/quat.",
            )

        posture_targets_2d: torch.Tensor | None = None
        if posture_targets is not None:
            posture_targets_2d = (
                posture_targets
                if posture_targets.ndim == 2
                else posture_targets.view(1, -1)
            )
            if posture_targets_2d.shape != q_init_2d.shape:
                raise ValueError(
                    f"{self.controller_name}: posture_targets must match q_init shape.",
                )

        hold_mask_1d: torch.Tensor | None = None
        if hold_mask is not None:
            hold_mask_1d = hold_mask.reshape(-1)
            if hold_mask_1d.shape[0] != batch_size:
                raise ValueError(f"{self.controller_name}: hold_mask batch mismatch.")

        return q_init_2d, pos_tgt_2d, quat_tgt_2d, posture_targets_2d, hold_mask_1d

    @staticmethod
    def _to_numpy_f64(x: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor to a float64 NumPy array without extra copies."""
        return x.detach().cpu().numpy().astype(np.float64, copy=False)

    def _solve_single_target(
        self,
        *,
        state: _MinkEnvState,
        q_init_row: torch.Tensor,
        pos_tgt_row: torch.Tensor,
        rot_tgt_row: torch.Tensor,
        iterations: int,
        integration_dt: float,
        posture_target_row: torch.Tensor | None,
        hold: bool,
    ) -> np.ndarray:
        """Solve one env target and return the controlled joint command."""
        q_init_np = self._to_numpy_f64(q_init_row)
        q_ctrl_init = q_init_np[self.dofs_idx]

        if hold:
            state.posture_target_q = (
                self._to_numpy_f64(posture_target_row)
                if posture_target_row is not None
                else q_init_np.copy()
            )
            return np.asarray(q_ctrl_init, dtype=np.float32)

        state.configuration.update(q=q_init_np)
        target = self._target_from_rows(pos_tgt_row=pos_tgt_row, rot_tgt_row=rot_tgt_row)
        state.position_task.set_target(target)
        state.orientation_task.set_target(target)

        if posture_target_row is not None:
            state.posture_target_q = self._to_numpy_f64(posture_target_row).copy()
        elif state.posture_target_q is None:
            state.posture_target_q = q_init_np.copy()
        state.posture_task.set_target(state.posture_target_q)

        try:
            self._iterate_single_target(
                state=state,
                iterations=iterations,
                integration_dt=integration_dt,
            )
            q_next_ctrl = state.configuration.q[self.dofs_idx].copy()
        except (mink.MinkError, ValueError):
            q_next_ctrl = q_ctrl_init.copy()

        return np.asarray(q_next_ctrl, dtype=np.float32)

    def _target_from_rows(
        self,
        *,
        pos_tgt_row: torch.Tensor,
        rot_tgt_row: torch.Tensor,
    ) -> mink.SE3:
        return mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3.from_matrix(self._to_numpy_f64(rot_tgt_row)),
            translation=self._to_numpy_f64(pos_tgt_row),
        )

    def _iterate_single_target(
        self,
        *,
        state: _MinkEnvState,
        iterations: int,
        integration_dt: float,
    ) -> None:
        for _ in range(iterations):
            vel = mink.solve_ik(
                state.configuration,
                state.tasks,
                integration_dt,
                solver=self.solver,
                damping=self.qp_damping,
                limits=state.limits,
                constraints=state.constraints,
            )
            state.configuration.integrate_inplace(vel, integration_dt)
            if self._target_reached(state):
                break

    def _target_reached(self, state: _MinkEnvState) -> bool:
        pos_err = state.position_task.compute_error(state.configuration)
        rot_err = state.orientation_task.compute_error(state.configuration)
        return (
            np.linalg.norm(pos_err[:3]) <= self.pos_tol
            and np.linalg.norm(rot_err[3:]) <= self.rot_tol
        )

    def solve_targets(
        self,
        *,
        q_init: torch.Tensor,
        pos_tgt: torch.Tensor,
        quat_tgt: torch.Tensor,
        iterations: int,
        integration_dt: float,
        posture_targets: torch.Tensor | None = None,
        hold_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve a batch of task-space targets into controlled joint positions.

        Args:
            q_init: Initial full-entity joint configuration.
            pos_tgt: Target positions in world-frame meters.
            quat_tgt: Target quaternions in WXYZ order.
            iterations: Maximum solver iterations.
            integration_dt: Integration time step in seconds.
            posture_targets: Posture targets used by the backend. Defaults to None.
            hold_mask: Boolean mask for environments that should hold position. Defaults to None.
        """
        if not self._states:
            raise RuntimeError(
                f"{self.controller_name}: backend must be build()'d before use.",
            )
        if iterations <= 0:
            raise ValueError(f"{self.controller_name}: iterations must be > 0.")
        if not np.isfinite(integration_dt) or integration_dt <= 0.0:
            raise ValueError(f"{self.controller_name}: integration_dt must be > 0.")

        q_init_2d, pos_tgt_2d, quat_tgt_2d, posture_targets_2d, hold_mask_1d = (
            self._normalize_solve_inputs(
                q_init=q_init,
                pos_tgt=pos_tgt,
                quat_tgt=quat_tgt,
                posture_targets=posture_targets,
                hold_mask=hold_mask,
            )
        )

        rot_tgt = quat_to_rotmat_wxyz_torch(quat_tgt_2d).contiguous()
        q_cmds: list[np.ndarray] = []

        for env_id, state in enumerate(self._states[: pos_tgt_2d.shape[0]]):
            q_cmds.append(
                self._solve_single_target(
                    state=state,
                    q_init_row=q_init_2d[env_id],
                    pos_tgt_row=pos_tgt_2d[env_id],
                    rot_tgt_row=rot_tgt[env_id],
                    iterations=iterations,
                    integration_dt=integration_dt,
                    posture_target_row=(
                        None
                        if posture_targets_2d is None
                        else posture_targets_2d[env_id]
                    ),
                    hold=(
                        False if hold_mask_1d is None else bool(hold_mask_1d[env_id])
                    ),
                ),
            )

        return torch.as_tensor(
            np.stack(q_cmds, axis=0),
            dtype=torch.float32,
            device=self.device,
        ).contiguous()

    def _make_state(self, model: mujoco.MjModel, end_link_name: str) -> _MinkEnvState:
        freeze_dofs = [idx for idx in range(model.nv) if idx not in self.dofs_idx]
        constraints: list[mink.Task] = []
        if freeze_dofs:
            constraints.append(
                mink.DofFreezingTask(model=model, dof_indices=freeze_dofs),
            )

        limits: list[mink.Limit] = [
            mink.ConfigurationLimit(
                model,
                gain=self.configuration_limit_gain,
            ),
        ]
        velocity_limits = self._build_velocity_limits(model)
        if velocity_limits is not None:
            limits.append(velocity_limits)

        configuration = mink.Configuration(model)
        position_task = mink.FrameTask(
            frame_name=end_link_name,
            frame_type=self.frame_type,
            position_cost=self.position_cost,
            orientation_cost=0.0,
            lm_damping=self.lm_damping,
        )
        orientation_task = mink.FrameTask(
            frame_name=end_link_name,
            frame_type=self.frame_type,
            position_cost=0.0,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
        )
        posture_task = mink.PostureTask(
            model=model,
            cost=self._resolve_posture_cost(model),
        )
        damping_task = mink.DampingTask(model=model, cost=self.damping_cost)
        return _MinkEnvState(
            configuration=configuration,
            position_task=position_task,
            orientation_task=orientation_task,
            posture_task=posture_task,
            damping_task=damping_task,
            limits=tuple(limits),
            constraints=tuple(constraints),
        )

    def _resolve_posture_cost(self, model: mujoco.MjModel) -> np.ndarray:
        if self.posture_cost.size == model.nv:
            return self.posture_cost.copy()

        if self.posture_cost.size != 1:
            raise ValueError(
                f"{self.controller_name}: posture_cost must be scalar or have length "
                "equal to model.nv.",
            )

        base_cost = float(
            np.asarray(self.posture_cost, dtype=np.float64).reshape(-1)[0],
        )
        cost = np.zeros((model.nv,), dtype=np.float64)
        cost[self.dofs_idx] = base_cost

        registry_name = str(getattr(self.model.__class__, "_registry_name", "") or "")
        if (
            registry_name in {"franka", "franka_with_gripper", "franka_fr3"}
            and len(self.dofs_idx) >= 7
        ):
            shoulder_indices = self.dofs_idx[:3]
            elbow_idx = self.dofs_idx[3]
            wrist_indices = self.dofs_idx[4:7]
            cost[shoulder_indices] = np.maximum(cost[shoulder_indices], 1e-2)
            cost[elbow_idx] = max(cost[elbow_idx], 5e-2)
            cost[wrist_indices] = np.maximum(cost[wrist_indices], 2e-1)

        return cost

    def _build_velocity_limits(
        self,
        model: mujoco.MjModel,
    ) -> mink.VelocityLimit | None:
        if self.max_joint_step is None:
            return None

        max_joint_velocity = self.max_joint_step / self.velocity_dt
        velocities: dict[str, np.ndarray] = {}
        for jid in range(model.njnt):
            joint_type = int(model.jnt_type[jid])
            if joint_type in (
                mujoco.mjtJoint.mjJNT_FREE,
                mujoco.mjtJoint.mjJNT_BALL,
            ):
                continue
            dof_adr = int(model.jnt_dofadr[jid])
            if dof_adr not in self.dofs_idx:
                continue
            velocities[model.joint(jid).name] = np.array(
                [max_joint_velocity],
                dtype=np.float64,
            )

        if not velocities:
            return None
        return mink.VelocityLimit(model, velocities=velocities)


__all__ = ["CostSpec", "MinkPoseBackend", "as_cost_array", "make_pose_backend"]
