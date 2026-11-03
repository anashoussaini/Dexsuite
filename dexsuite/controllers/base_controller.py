"""Provide base classes and utilities for DexSuite controllers.

Controllers in DexSuite are lightweight wrappers that map an action tensor to
low-level simulator commands (position, velocity, or force) for a subset of a
robot's degrees of freedom (DoFs).

This module defines the Controller base class, which provides:

- Shape handling for single vs. batched environments.
- Optional gain and limit installation from the model.
- Convenience helpers for forward kinematics and inverse kinematics.

The controller API is intentionally small: environments and robots construct a
controller, call post_build once after the scene is built, and then call step at
each control cycle.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch

from dexsuite.utils.globals import get_device, get_n_envs

if TYPE_CHECKING:
    from collections.abc import Sequence


class Controller(abc.ABC):
    """Represent the shared interface for DexSuite robot controllers.

    A controller transforms an action vector into simulator commands applied to
    a subset of a robot's degrees of freedom (DoFs). Subclasses implement
    action_space and _apply_cmd.

    DexSuite supports vectorized environments. In a single environment actions
    may be unbatched `(D,)` tensors, while vectorized scenes use `(n_envs, D)`
    tensors. The associated model object must define `dof` and may define
    control DoF indices, gains, force ranges, home poses, and end-effector links.

    Attributes:
        entity: Simulator entity controlled by this controller.
        model: Robot component model associated with the entity.
        dofs_idx: Local DoF indices controlled by this controller.
        n_envs: Number of parallel environments in the scene.
        device: Torch device used for controller tensors.
        kp: Optional proportional gains (torch tensor on device).
        kv: Optional derivative gains (torch tensor on device).
        force_range: Optional symmetric force/torque limits.
        home_q: Optional home configuration for controlled DoFs.
    """

    def __init__(
        self,
        *,
        entity: Any,
        model: Any,
        **_: Any,
    ) -> None:
        """Initialize the object.

        Args:
        entity: The simulation entity to control (Genesis entity-like API).
        model: The robot component model.
        _: Ignored extra keyword arguments (kept for forward compatibility).
        """
        self.entity = entity
        self.model = model

        control_dofs_index = getattr(self.model, "control_dofs_index", None)
        if control_dofs_index is None:
            self.dofs_idx: list[int] = list(range(int(model.dof)))
        else:
            self.dofs_idx = list(control_dofs_index)

        self.n_envs: int = get_n_envs()
        self._batched: bool = self.n_envs > 1
        self.device: torch.device = get_device()

        self.kp = self._load_control_param(getattr(model, "kp", None), "kp")
        self.kv = self._load_control_param(getattr(model, "kv", None), "kv")
        self.force_range = self._load_control_param(
            getattr(model, "force_range", None),
            "force_range",
        )

        home_q = getattr(self.model, "home_q", None)
        self.home_q: torch.Tensor | None = (
            torch.as_tensor(home_q, dtype=torch.float32, device=self.device)
            .contiguous()
            .view(-1)
            if home_q is not None
            else None
        )
        self._home_q_row: torch.Tensor | None = (
            self.home_q.view(1, -1).contiguous() if self.home_q is not None else None
        )

    def _load_control_param(
        self,
        value: Any,
        name: str,
    ) -> torch.Tensor | None:
        """Load a controller parameter for the active DoF slice.

        Parameters can be provided either for the controlled DoFs only or for
        the full robot DoFs. Full-DoF vectors are sliced down to the
        controller's active DoFs.
        """
        if value is None:
            return None

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        tensor = tensor.reshape(-1).contiguous()

        if tensor.numel() == 1 and self.dof > 1:
            return tensor.expand(self.dof).contiguous()
        if tensor.numel() == self.dof:
            return tensor

        model_dof = getattr(self.model, "dof", None)
        if model_dof is not None and tensor.numel() == int(model_dof):
            return tensor[self.dofs_idx].contiguous()

        raise ValueError(
            f"{name} length {tensor.numel()} does not match controlled DoFs "
            f"{self.dof} or model DoFs {model_dof}.",
        )

    @property
    def dof(self) -> int:
        """Return the number of controlled degrees of freedom."""
        return len(self.dofs_idx)

    def _expect_2d(self, x: torch.Tensor, cols: int) -> torch.Tensor:
        """Validate and reshape an input tensor to 2D (batch, cols).

        Args:
            x: Input tensor, either 1D (cols,) (single env) or 2D (n_envs, cols) (batched envs).
            cols: Expected size of the last dimension.

        Returns:
            A 2D tensor of shape (1, cols) (single env) or (n_envs, cols)
            (batched envs).

        Raises:
            TypeError: If x is not a torch tensor.
            ValueError: If the tensor has the wrong rank or shape.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input must be a torch.Tensor.")
        if x.ndim == 1:
            if self.n_envs != 1:
                raise ValueError(
                    f"Expected input shape ({self.n_envs}, {cols}) in batched mode; "
                    f"got 1D tensor.",
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
        """Convert a batched tensor to the shape expected by Genesis.

        Genesis APIs generally expect (D,) for single-env scenes and (n_envs, D)
        for batched scenes. This helper performs that squeeze when n_envs == 1.
        """
        return x2d[0] if not self._batched else x2d

    def _get_end_effector_link(self) -> Any:
        """Return the end-effector link handle defined by model.end_link."""
        link_name = getattr(self.model, "end_link", None)
        if not link_name:
            raise ValueError("model must define end_link.")
        return self.entity.get_link(link_name)

    def _fk_pos_quat_2d(self, link_idx_local: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward kinematics for a single link and return batched outputs.

        Args:
            link_idx_local: Link index local to the simulation entity.

        Returns:
            Tuple (pos, quat) where pos has shape (N, 3) and quat has shape
            (N, 4) in wxyz convention, with N == n_envs for batched environments
            and N == 1 otherwise.
        """
        qpos = self.entity.get_dofs_position()
        if self._batched and qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).expand(self.n_envs, -1)
        qpos = qpos.contiguous()

        pos, quat = self.entity.forward_kinematics(
            qpos=qpos,
            links_idx_local=[link_idx_local],
        )
        pos = pos.reshape(-1, 3).to(self.device).contiguous()
        quat = quat.reshape(-1, 4).to(self.device).contiguous()
        return pos, quat

    @staticmethod
    def _reduce_bounds_1d(
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce batched DoF bounds to 1D bounds.

        Some Genesis APIs return per-env bounds with shape (N, D). This helper
        reduces them to 1D bounds with shape (D,) using min and max across
        environments.
        """
        if lo.ndim == 2:
            lo = lo.min(dim=0).values
        if hi.ndim == 2:
            hi = hi.max(dim=0).values
        return lo, hi

    def _env_indices(
        self,
        env_idx: torch.Tensor | Sequence[int] | None,
    ) -> torch.Tensor | None:
        """Normalize optional environment indices for controller state tensors."""
        if env_idx is None:
            return None
        indices = torch.as_tensor(env_idx, dtype=torch.long, device=self.device).view(
            -1
        )
        if not ((indices >= 0) & (indices < self.n_envs)).all().item():
            raise ValueError(
                f"env_idx {indices.detach().cpu().tolist()} out of range "
                f"[0, {self.n_envs}).",
            )
        return indices

    def post_build(self) -> None:
        """Finalize controller construction after the scene is built.

        Subclasses may override this to fetch simulator-derived limits or cache
        handles. The base implementation installs runtime PD gains/limits onto
        the simulator entity (if provided by the model).
        """
        self.install_runtime_gains()

    def reset(self, env_idx: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset any controller-internal state.

        Most controllers are stateless and do not need to override this hook.
        Controllers with internal solver caches or reference targets can
        override it and robot reset paths will call it.

        Args:
            env_idx: Optional subset of environment indices to reset.
        """
        del env_idx

    def install_runtime_gains(self) -> None:
        """Install PD gains and force limits on the simulation entity.

        This is a best-effort operation: if the model does not define a given
        attribute (kp, kv, force_range), that part is skipped.
        """
        if self.kp is not None:
            self.entity.set_dofs_kp(self.kp, self.dofs_idx)
        if self.kv is not None:
            self.entity.set_dofs_kv(self.kv, self.dofs_idx)
        if self.force_range is not None:
            lo = -self.force_range
            hi = self.force_range
            self.entity.set_dofs_force_range(lo, hi, self.dofs_idx)

    def _set_home_position_state_only(
        self,
        env_idx: torch.Tensor | None = None,
    ) -> None:
        """Reset controlled joints to the model's home configuration.

        If the model does not define home_q, this method uses zeros for the
        controlled degrees of freedom.

        Broadcasting rules:

        - If home_q contains one scalar and the controller drives multiple DoFs,
          it is broadcast to all controlled DoFs.
        - Otherwise, home_q must match the number of controlled DoFs.

        Genesis expects a 1D position vector when n_envs == 1 and env_idx is
        None, so this method squeezes the leading batch dimension in that case.

        Args:
            env_idx: Optional subset of environment indices to reset.

        Raises:
            ValueError: If home_q is defined but does not match the number of
                controlled DoFs.
        """
        ctrl_dof = len(self.dofs_idx)
        if self.home_q is None:
            q_row = torch.zeros((1, ctrl_dof), dtype=torch.float32, device=self.device)
        else:
            q = self.home_q.reshape(-1)
            if q.numel() == 1 and ctrl_dof > 1:
                q = q.expand(ctrl_dof)
            if q.numel() != ctrl_dof:
                raise ValueError(
                    f"home_q length {q.numel()} != controlled DoFs {ctrl_dof}.",
                )
            q_row = q.view(1, ctrl_dof)

        n_reset = env_idx.numel() if env_idx is not None else self.n_envs
        q_target = q_row.repeat(n_reset, 1).contiguous()
        if self.n_envs == 1 and env_idx is None:
            pos_to_set = q_target.squeeze(0)
        else:
            pos_to_set = q_target

        self.entity.set_dofs_position(
            pos_to_set,
            dofs_idx_local=self.dofs_idx,
            envs_idx=env_idx,
            zero_velocity=True,
        )

    def set_home_position(self, env_idx: torch.Tensor | None = None) -> None:
        """Reset controlled joints and command the home configuration.

                Most public controllers keep the simulator's position target aligned
                with the reset state. Pose controllers using the Genesis internal
                backend call _set_home_position_state_only directly to preserve that
                backend's state-only reset behavior.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        self._set_home_position_state_only(env_idx=env_idx)

        ctrl_dof = len(self.dofs_idx)
        if self.home_q is None:
            q_row = torch.zeros((1, ctrl_dof), dtype=torch.float32, device=self.device)
        else:
            q = self.home_q.reshape(-1)
            if q.numel() == 1 and ctrl_dof > 1:
                q = q.expand(ctrl_dof)
            if q.numel() != ctrl_dof:
                raise ValueError(
                    f"home_q length {q.numel()} != controlled DoFs {ctrl_dof}.",
                )
            q_row = q.view(1, ctrl_dof)

        n_reset = env_idx.numel() if env_idx is not None else self.n_envs
        q_target = q_row.repeat(n_reset, 1).contiguous()
        if self.n_envs == 1 and env_idx is None:
            pos_to_set = q_target.squeeze(0)
        else:
            pos_to_set = q_target

        self.entity.control_dofs_position(
            pos_to_set,
            dofs_idx_local=self.dofs_idx,
            envs_idx=env_idx,
        )

    def _full_qpos_template(self, batch: int) -> torch.Tensor:
        """Return a full-DoF joint-position template.

        Args:
            batch: Batch size to match the IK target batch.

        Returns:
            A contiguous clone of the entity's joint positions with shape
            (batch, dof_total).
        """
        qpos = self.entity.get_dofs_position()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[0] != batch:
            qpos = qpos.expand(batch, -1)
        return qpos.to(self.device).contiguous().clone()

    def _control_position(self, target: torch.Tensor) -> None:
        """Send a joint position command to the entity.

        Args:
            target: Desired joint positions for the controlled DoFs, shaped (D,) or (n_envs, D).
        """
        target = self._expect_2d(target, self.dof)
        self.entity.control_dofs_position(self._to_entity_shape(target), self.dofs_idx)

    def _control_velocity(self, velocity: torch.Tensor) -> None:
        """Send a joint velocity command to the entity.

        Args:
            velocity: Desired joint velocities for the controlled DoFs, shaped (D,) or (n_envs, D).
        """
        velocity = self._expect_2d(velocity, self.dof)
        self.entity.control_dofs_velocity(
            self._to_entity_shape(velocity),
            self.dofs_idx,
        )

    def _control_force(self, force_2d: torch.Tensor) -> None:
        """Send a joint force/torque command to the entity.

        Args:
            force_2d: Desired forces/torques for the controlled DoFs, shaped (D,) or (n_envs, D).
        """
        tau = self._expect_2d(force_2d, self.dof)
        self.entity.control_dofs_force(self._to_entity_shape(tau), self.dofs_idx)

    @abc.abstractmethod
    def action_space(self) -> Any:
        """Return this controller's action space.

        Returns:
            A Gymnasium spaces.Box describing the action bounds and shape.
        """

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> None:
        """Apply a control action.

        Subclasses implement `_apply_cmd` and can assume the command follows
        the shape conventions enforced by `_expect_2d`.

        Args:
            action: Action tensor with shape (D,) (single env) or (n_envs, D) (batched envs).
        """
        self._apply_cmd(action)

    @abc.abstractmethod
    def _apply_cmd(self, vec: torch.Tensor) -> None:
        """Apply the processed control command vector.

        Args:
            vec: Action tensor, either (D,) or (n_envs, D) depending on the scene configuration.
        """
