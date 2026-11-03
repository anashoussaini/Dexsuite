"""Define a robot whose manipulator and gripper share one entity."""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch
from gymnasium import spaces

from dexsuite.controllers import make as make_ctrl
from dexsuite.core.components.arm import IntegratedManipulatorModel
from dexsuite.options import ControllerOptions
from dexsuite.utils import get_device
from dexsuite.utils.action_utils import build_single_layout, dispatch
from dexsuite.utils.robot_utils import (
    attach_tcp_anchor,
    get_end_effector_pose,
    get_robot_morph,
    get_tool_center_point_pose,
)
from dexsuite.utils.tactile_utils import RigidPadTactile


class _HandProxy:
    """Proxy built-in gripper metadata for integrated robot code paths."""

    def __init__(
        self,
        manipulator: IntegratedManipulatorModel,
        device: torch.device,
    ):
        """Initialize the object.

        Args:
        manipulator: Integrated manipulator model that exposes built-in gripper metadata.
        device: Torch device used for proxy tensors.
        """
        self.dof = int(getattr(manipulator, "builtin_gripper_dof", 0))
        self.device = device
        self.root_link = getattr(
            manipulator,
            "builtin_gripper_root_link",
            manipulator.end_link,
        )
        self.control_dofs_index = list(manipulator.gripper_dofs_index)
        self.tcp_pose = getattr(manipulator, "builtin_gripper_tcp_pose", None)

        self.home_q = getattr(manipulator, "builtin_gripper_home_q", None)
        self.q_lo = getattr(manipulator, "builtin_gripper_q_lo", None)
        self.q_hi = getattr(manipulator, "builtin_gripper_q_hi", None)
        self.kp = self._slice_joint_param(manipulator, "kp")
        self.kv = self._slice_joint_param(manipulator, "kv")
        self.force_range = self._slice_joint_param(manipulator, "force_range")

    def _slice_joint_param(
        self,
        manipulator: IntegratedManipulatorModel,
        name: str,
    ) -> torch.Tensor | None:
        value = getattr(manipulator, name, None)
        if value is None:
            return None

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        tensor = tensor.reshape(-1).contiguous()
        if tensor.numel() == 1 and self.dof > 1:
            return tensor.expand(self.dof).contiguous()
        if tensor.numel() == self.dof:
            return tensor
        if tensor.numel() != int(manipulator.dof):
            raise ValueError(
                f"{name} length {tensor.numel()} does not match gripper DoFs "
                f"{self.dof} or manipulator DoFs {manipulator.dof}.",
            )
        return tensor[self.control_dofs_index].contiguous()


class IntegratedRobot:
    """Represent a single-entity robot with a built-in gripper.

    Arm commands address the manipulator DoFs, while gripper commands address explicit full-entity gripper DoF indices. Observations expose arm and hand state separately even though Genesis stores them in one entity.

    Attributes:
        entity: Genesis entity containing the arm and gripper.
        arm_model: Integrated manipulator model.
        hand_model: Proxy exposing built-in gripper metadata.
        arm_controller: Controller for manipulator DoFs.
        gripper_controller: Controller for gripper DoFs.
        layout: Flat action layout for arm and gripper commands.
    """

    def __init__(
        self,
        *,
        manipulator: IntegratedManipulatorModel,
        arm_control_options: ControllerOptions,
        gripper_control_options: ControllerOptions | None,
        scene: gs.Scene,
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        visualize_tcp: bool = False,
    ):
        """Initialize the object.

        Args:
        manipulator: Integrated manipulator model instance.
        arm_control_options: Configuration for the arm controller.
        gripper_control_options: Configuration for the gripper controller.
        scene: Genesis scene that owns the robot entity.
        base_pos: Robot base position `[x, y, z]` in world meters.
        quat: Robot base orientation `[w, x, y, z]` in world frame.
        visualize_tcp: Whether to visualize the tool center point. Defaults to False.
        """
        self.manip = manipulator
        self.scene = scene
        self.device = get_device()

        self.arm_ent = scene.add_entity(
            get_robot_morph(manipulator, pos=base_pos, quat=quat),
            material=gs.materials.Rigid(needs_coup=True, gravity_compensation=1.0),
            visualize_contact=bool(getattr(manipulator, "visualize_contact", False)),
        )
        self._link_names = [link.name for link in self.arm_ent.links]
        self._eef_link_idx = self._link_names.index(self.manip.end_link)

        self.gripper = _HandProxy(self.manip, device=self.device)
        self.hand_ent = self.arm_ent
        self.arm_ctrl = make_ctrl(
            arm_control_options,
            entity=self.arm_ent,
            model=self.manip,
        )

        self.hand_ctrl = None
        if gripper_control_options:
            self.hand_ctrl = make_ctrl(
                gripper_control_options,
                entity=self.arm_ent,
                model=self.gripper,
            )

        self.dof = int(self.manip.dof)
        self.tactile: RigidPadTactile | None = None

        attach_tcp_anchor(self, visualize_tcp)

    @property
    def action_space(self) -> spaces.Box:
        """Get the robot's flat Gymnasium action space."""
        return self._layout.as_box()

    def apply_action(self, action) -> None:
        """Apply an action to the robot's controllers.

        Args:
            action: Action tensor to validate or dispatch.
        """
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
        """Initialize the PD controllers for the arm and gripper."""
        self.arm_ctrl.post_build()
        if self.hand_ctrl:
            self.hand_ctrl.post_build()
        tactile_enabled = bool(getattr(self.manip, "tactile", False))
        tactile_links = tuple(getattr(self.manip, "builtin_tactile_links", ()) or ())
        if tactile_enabled and tactile_links:
            self.tactile = RigidPadTactile(
                self.hand_ent,
                link_names=tactile_links,
                grid_size=getattr(self.manip, "builtin_tactile_grid_size", None),
                point_distance=float(
                    getattr(self.manip, "builtin_tactile_point_distance", 0.002)
                    or 0.002,
                ),
            )
        self._layout = build_single_layout(
            arm_ctrl=self.arm_ctrl,
            grip_ctrl=self.hand_ctrl,
        )
        self._act_dim = self._layout.total_dim

    def reset(self, env_idx=None) -> None:
        """Reset the robot to its home joint configuration.

        Args:
            env_idx: Environment indices to reset or sample. Defaults to None.
        """
        self.arm_ctrl.reset(env_idx=env_idx)
        self.arm_ctrl.set_home_position(env_idx=env_idx)
        if self.hand_ctrl is not None:
            self.hand_ctrl.reset(env_idx=env_idx)
            self.hand_ctrl.set_home_position(env_idx=env_idx)

    def get_obs(self) -> dict[str, Any]:
        """Retrieves the observation dictionary for the robot."""
        qpos_full = self.arm_ent.get_dofs_position()
        qvel_full = self.arm_ent.get_dofs_velocity()
        qtorque_full = self.arm_ent.get_dofs_force()

        eef_pos, eef_quat = get_end_effector_pose(
            self.arm_ent,
            self._eef_link_idx,
            qpos=qpos_full,
        )
        manipulator_observation = {
            "qpos": qpos_full,
            "qvel": qvel_full,
            "qtorque": qtorque_full,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
        }

        gripper_dofs = list(self.manip.gripper_dofs_index)
        tcp_pos, tcp_quat = get_tool_center_point_pose(self)
        gripper_observation = {
            "qpos": qpos_full[..., gripper_dofs],
            "qvel": qvel_full[..., gripper_dofs],
            "qtorque": qtorque_full[..., gripper_dofs],
            "tcp_pos": tcp_pos,
            "tcp_quat": tcp_quat,
        }
        if self.tactile is not None:
            gripper_observation["tactile"] = self.tactile.read()

        return {"manipulator": manipulator_observation, "gripper": gripper_observation}
