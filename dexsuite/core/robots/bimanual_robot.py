"""Define a bimanual robot wrapper from two single-arm robots."""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch
from gymnasium import spaces

from dexsuite.core.robots.factory import make_arm_from_options
from dexsuite.options import ArmOptions, PoseOptions
from dexsuite.utils.action_utils import build_bimanual_layout, dispatch


class BimanualRobot:
    """Represent a bimanual robot composed of left and right arms.

    Actions are flattened with left-arm controls first, then right-arm controls. Observations are returned under separate `left` and `right` keys to preserve arm ownership.

    Attributes:
        left: Left single-arm robot wrapper.
        right: Right single-arm robot wrapper.
        layout: Flat action layout shared by both arms.
    """

    def __init__(
        self,
        *,
        arm_options: tuple[ArmOptions, ArmOptions],
        scene: gs.Scene,
        pose_options: tuple[PoseOptions, PoseOptions],
        visualize_tcp: bool = False,
    ):
        """Initialize the object.

        Args:
        arm_options: Pair of left and right arm option objects.
        scene: Genesis scene to which both robots are added.
        pose_options: Pair of left and right base pose option objects.
        visualize_tcp: Whether to visualize tool center points. Defaults to False.
        """
        self.left = make_arm_from_options(
            arm_options=arm_options[0],
            scene=scene,
            pose_options=pose_options[0],
            visualize_tcp=visualize_tcp,
        )
        self.right = make_arm_from_options(
            arm_options=arm_options[1],
            scene=scene,
            pose_options=pose_options[1],
            visualize_tcp=visualize_tcp,
        )

    @property
    def action_space(self) -> spaces.Box:
        """Get the robot's flat Gymnasium action space."""
        return self._layout.as_box()

    def apply_action(self, action) -> None:
        """Validate and dispatch a flat action tensor to both arms.

        Args:
            action: Action tensor to validate or dispatch.
        """
        if not torch.is_tensor(action):
            raise TypeError("action must be a torch.Tensor")
        if action.dtype != torch.float32:
            raise TypeError("action dtype must be float32")

        if action.ndim == 1:
            if action.numel() != self._act_dim:
                raise ValueError(
                    f"Expected ({self._act_dim},), got {tuple(action.shape)}",
                )
        elif action.ndim == 2:
            if action.shape[1] != self._act_dim:
                raise ValueError(
                    f"Expected (_, {self._act_dim}), got {tuple(action.shape)}",
                )
        else:
            raise ValueError("Action must be 1D or 2D.")
        dispatch(self._layout, action)

    def apply_action_validated(self, action: torch.Tensor) -> None:
        """Apply an action without re-validating shape/dtype.

                Intended for hot paths that already validated the flat action at the
                environment boundary.

        Args:
            action: Action tensor to validate or dispatch.
        """
        dispatch(self._layout, action)

    def install_pd(self) -> None:
        """Initialize the PD controllers for both the left and right arms."""
        self.left.install_pd()
        self.right.install_pd()
        left_arm = self.left.arm_ctrl
        left_grip = getattr(self.left, "hand_ctrl", None)
        right_arm = self.right.arm_ctrl
        right_grip = getattr(self.right, "hand_ctrl", None)
        self._layout = build_bimanual_layout(
            left_arm=left_arm,
            left_grip=left_grip,
            right_arm=right_arm,
            right_grip=right_grip,
        )

        self._act_dim = self._layout.total_dim

    def reset(self, env_idx=None) -> None:
        """Reset both the left and right arms to their home positions.

        Args:
            env_idx: Optional indices of environments to reset. If None, resets all.
        """
        self.left.reset(env_idx=env_idx)
        self.right.reset(env_idx=env_idx)

    def get_obs(self) -> dict[str, Any]:
        """Retrieve the combined observations from both arms.

        Returns:
            A dictionary containing the observations for the 'left' and 'right'
            arms.
        """
        return {"left": self.left.get_obs(), "right": self.right.get_obs()}
