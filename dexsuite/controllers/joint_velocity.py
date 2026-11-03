"""Provide joint-space velocity control.

This controller commands target joint velocities for a subset of the
entity's degrees of freedom (DoFs).

Two action conventions are supported:

1. Normalized: actions are in [-1, 1] and scaled by v_max.
2. Unnormalized: actions are interpreted directly as velocities in each
   joint's native units per second (typically rad/s for revolute joints).
"""

from __future__ import annotations

import math

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller


@register_controller("joint_velocity")
class JointVelocityController(Controller):
    """Control robot joints with velocity commands.

    Action semantics:

    - If normalized is True:
      - Input action a is optionally clipped to [-1, 1] when clip is True.
      - The commanded velocity is the (possibly clipped) action scaled by v_max.

    - If normalized is False:
      - The action is interpreted directly as a velocity target.
      - When clip is True, velocities are clamped to [-v_max, v_max].

    `v_max` is required for both modes to keep the action space bounded and
    predictable.
    """

    def __init__(
        self,
        *,
        entity: object,
        model: object,
        normalized: bool | None,
        clip: bool | None,
        v_max: float | None,
        **kw: object,
    ) -> None:
        """Initialize the object.

        Args:
            entity: The simulation entity to control.
            model: The corresponding robot component model.
            normalized: Whether actions are in the normalized range [-1, 1].
            clip: Whether to clip actions to the valid range.
            v_max: The maximum velocity for normalized actions.
            kw: Additional keyword arguments for the base controller.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        super().__init__(entity=entity, model=model, **kw)
        if normalized is None or clip is None or v_max is None:
            raise ValueError(
                "joint_velocity: 'normalized', 'clip', and 'v_max' are required.",
            )
        if not (
            isinstance(v_max, int | float)
            and math.isfinite(float(v_max))
            and v_max > 0.0
        ):
            raise ValueError(f"joint_velocity: v_max must be > 0, got {v_max!r}.")
        self.normalized = bool(normalized)
        self.clip = bool(clip)
        self.v_max = float(v_max)

    def action_space(self) -> spaces.Box:
        """Return the controller's Gymnasium action space.

        Returns:
            A spaces.Box of shape (dof,).

            - If normalized is True, the space is always [-1, 1].
            - If normalized is False, the space is [-v_max, v_max] in
              joint velocity units.
        """
        if self.normalized:
            return float_box(-1.0, 1.0, shape=(self.dof,))
        return float_box(-self.v_max, self.v_max, shape=(self.dof,))

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        """Apply a joint velocity command.

        Args:
            vec_2d: Action tensor with shape (D,) or (n_envs, D).
        """
        a = self._expect_2d(vec_2d, self.dof).contiguous()
        if self.normalized:
            v = (a.clamp(-1.0, 1.0) if self.clip else a) * self.v_max
        else:
            v = a.clamp(-self.v_max, self.v_max) if self.clip else a
        self._control_velocity(v.contiguous())
