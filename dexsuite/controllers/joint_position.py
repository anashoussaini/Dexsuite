"""Provide joint-space position control.

This controller commands target joint positions for a subset of the
entity's degrees of freedom (DoFs).

Two action conventions are supported:

1. Normalized (recommended for RL): actions are in [-1, 1] and mapped
   linearly into the joint limits read from the simulator.
2. Unnormalized (metric): actions are interpreted as absolute joint targets
   in each joint's native units (typically radians for revolute joints).
"""

from __future__ import annotations

import torch
from gymnasium import spaces

from dexsuite.core.registry import register_controller
from dexsuite.utils.space_utils import float_box

from .base_controller import Controller


@register_controller("joint_position")
class JointPositionController(Controller):
    """Control robot joints with position commands.

    Action semantics:

    - If normalized is True:
      - Input action a is optionally clipped to [-1, 1] when clip is True.
      - The action is mapped per DoF linearly from [-1, 1] to the simulator
        limits [q_lo, q_hi].

    - If normalized is False:
      - The action is interpreted directly as absolute joint targets.
      - If clip is True and joint limits are available, targets are clamped to
        [q_lo, q_hi].

    Joint limits are fetched from the simulator in `post_build`, so
    non-normalized action spaces and normalized mapping require `post_build`
    before stepping.
    """

    def __init__(
        self,
        *,
        entity: object,
        model: object,
        normalized: bool | None,
        clip: bool | None,
        **kw: object,
    ) -> None:
        """Initialize the object.

        Args:
            entity: The simulation entity to control.
            model: The corresponding robot component model.
            normalized: Whether actions are in the normalized range [-1, 1].
            clip: Whether to clip actions to the valid range.
            kw: Additional keyword arguments for the base controller.

        Raises:
            ValueError: If required parameters are missing.
        """
        super().__init__(entity=entity, model=model, **kw)
        if normalized is None or clip is None:
            raise ValueError("joint_position: 'normalized' and 'clip' are required.")
        self.normalized = bool(normalized)
        self.clip = bool(clip)
        self._lo_1d: torch.Tensor | None = None
        self._hi_1d: torch.Tensor | None = None

    def post_build(self) -> None:
        """Fetch joint limits from the simulator.

        Genesis may report limits per environment ((n_envs, D)). This method
        reduces them to a single pair of 1D bound vectors (D,) using min and max
        across environments.
        """
        super().post_build()
        lo, hi = self.entity.get_dofs_limit(dofs_idx_local=self.dofs_idx)
        lo, hi = self._reduce_bounds_1d(lo, hi)
        self._lo_1d = lo.to(dtype=torch.float32, device=self.device).contiguous()
        self._hi_1d = hi.to(dtype=torch.float32, device=self.device).contiguous()

    def action_space(self) -> spaces.Box:
        """Return the controller's Gymnasium action space.

        Returns:
            A spaces.Box of shape (dof,).

            - If normalized is True, the space is always [-1, 1].
            - If normalized is False, the space bounds are the simulator joint
              limits. This requires post_build to have been called.

        Raises:
            RuntimeError: If limits are required but not initialized yet.
        """
        if self.normalized:
            return float_box(-1.0, 1.0, shape=(self.dof,))
        if self._lo_1d is None or self._hi_1d is None:
            raise RuntimeError(
                "joint_position.action_space (non-normalized) requires built limits. "
                "Call env.reset() after ds.make() before querying action_space().",
            )
        return float_box(self._lo_1d, self._hi_1d)

    @torch.no_grad()
    def _apply_cmd(self, vec_2d: torch.Tensor) -> None:
        """Apply a joint position command.

        Args:
            vec_2d: Action tensor with shape (D,) or (n_envs, D).

        Raises:
            RuntimeError: If normalized mapping is requested but limits are not
                initialized (missing post_build).
        """
        a = self._expect_2d(vec_2d, self.dof).contiguous()
        if self.normalized:
            if self._lo_1d is None or self._hi_1d is None:
                raise RuntimeError(
                    "joint_position: limits not initialized; call env.reset() first.",
                )
            a = a.clamp(-1.0, 1.0) if self.clip else a
            tgt = (
                self._lo_1d + (a + 1.0) * 0.5 * (self._hi_1d - self._lo_1d)
            ).contiguous()
        else:
            tgt = a
            if self.clip and (self._lo_1d is not None) and (self._hi_1d is not None):
                tgt = torch.minimum(
                    torch.maximum(tgt, self._lo_1d),
                    self._hi_1d,
                ).contiguous()
        self._control_position(tgt)
