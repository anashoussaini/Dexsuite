"""Provide joint-space torque and force control.

This controller commands direct joint torques and forces for a subset of the
entity's degrees of freedom (DoFs).

Two action conventions are supported:

1. Normalized: actions are in [-1, 1] and scaled by tau_max.
2. Unnormalized: actions are interpreted directly as torques/forces in the
   simulator's native units (typically N·m for revolute joints).
"""

from __future__ import annotations

import math

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller


@register_controller("joint_torque")
class JointTorqueController(Controller):
    """Control robot joints with torque or force commands.

    Action semantics:

    - If normalized is True:
      - Input action a is optionally clipped to [-1, 1] when clip is True.
      - The commanded torque is the (possibly clipped) action scaled by tau_max.

    - If normalized is False:
      - The action is interpreted directly as the commanded torque or force.
      - When clip is True and simulator limits are available, torques are
        clamped to those limits.

    `tau_max` is required for both modes to keep configuration consistent and
    to provide bounded action spaces when normalized. When `normalized` is
    false, the action space is taken from simulator limits fetched in `post_build`.
    """

    def __init__(
        self,
        *,
        entity: object,
        model: object,
        normalized: bool | None,
        clip: bool | None,
        tau_max: float | None,  # used when normalized=True
        **kw: object,
    ) -> None:
        """Initialize the object.

        Args:
            entity: The simulation entity to control.
            model: The corresponding robot component model.
            normalized: Whether actions are in the normalized range [-1, 1].
            clip: Whether to clip actions to the valid range.
            tau_max: The maximum torque for normalized actions.
            kw: Additional keyword arguments for the base controller.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        super().__init__(entity=entity, model=model, **kw)
        if normalized is None or clip is None or tau_max is None:
            raise ValueError(
                "joint_torque: 'normalized', 'clip', and 'tau_max' are required.",
            )
        if not (
            isinstance(tau_max, int | float)
            and math.isfinite(float(tau_max))
            and tau_max > 0.0
        ):
            raise ValueError(f"joint_torque: tau_max must be > 0, got {tau_max!r}.")
        self.normalized = bool(normalized)
        self.clip = bool(clip)
        self.tau_max = float(tau_max)

        self._lo_tau: torch.Tensor | None = None
        self._hi_tau: torch.Tensor | None = None

    def post_build(self) -> None:
        """Fetch torque/force limits from the simulator.

        Genesis may report per-environment limits with shape (n_envs, D). This
        method reduces them to 1D bounds (D,) using min and max across
        environments.
        """
        super().post_build()
        lo, hi = self.entity.get_dofs_force_range(dofs_idx_local=self.dofs_idx)
        lo, hi = self._reduce_bounds_1d(lo, hi)
        self._lo_tau = lo.to(dtype=torch.float32, device=self.device).contiguous()
        self._hi_tau = hi.to(dtype=torch.float32, device=self.device).contiguous()

    def action_space(self) -> spaces.Box:
        """Return the controller's Gymnasium action space.

        Returns:
            A spaces.Box of shape (dof,).

            - If normalized is True, the space is always [-1, 1].
            - If normalized is False, the space bounds are the simulator
              torque and force limits. This requires post_build to have been
              called.

        Raises:
            RuntimeError: If limits are required but not initialized yet.
        """
        if self.normalized:
            return float_box(-1.0, 1.0, shape=(self.dof,))
        if self._lo_tau is None or self._hi_tau is None:
            raise RuntimeError(
                "joint_torque.action_space (non-normalized) requires built limits. "
                "Call env.reset() after ds.make() before querying action_space().",
            )
        return float_box(self._lo_tau, self._hi_tau)

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        """Apply a joint torque/force command.

        Args:
            vec_2d: Action tensor with shape (D,) or (n_envs, D).

        Raises:
            RuntimeError: If non-normalized clipping requires limits but limits
                are not initialized (missing post_build).
        """
        a = self._expect_2d(vec_2d, self.dof).contiguous()
        if self.normalized:
            tau = (a.clamp(-1.0, 1.0) if self.clip else a) * self.tau_max
        else:
            if self._lo_tau is None or self._hi_tau is None:
                raise RuntimeError(
                    "joint_torque: limits not initialized; call env.reset() first.",
                )
            tau = (
                torch.minimum(torch.maximum(a, self._lo_tau), self._hi_tau)
                if self.clip
                else a
            )
        self._control_force(tau.contiguous())
