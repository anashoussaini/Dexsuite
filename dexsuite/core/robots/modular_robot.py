"""Define a robot built from a separate manipulator and gripper."""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch
from gymnasium import spaces

from dexsuite.controllers import make as make_ctrl
from dexsuite.core.components.arm import ModularManipulatorModel
from dexsuite.core.components.gripper import GripperModel
from dexsuite.core.components.mount import GripperMount
from dexsuite.options import ControllerOptions
from dexsuite.utils import get_device
from dexsuite.utils.action_utils import build_single_layout, dispatch
from dexsuite.utils.robot_utils import (
    attach_tcp_anchor,
    create_mount_morph,
    get_end_effector_pose,
    get_robot_morph,
    get_tool_center_point_pose,
    link_entities,
)
from dexsuite.utils.tactile_utils import RigidPadTactile


class ModularRobot:
    """Represent a modular robot with a mounted gripper.

    The manipulator and gripper are separate Genesis entities linked by a fixed mount after construction. Flat actions are dispatched through the shared action layout.

    Attributes:
        arm_entity: Genesis entity for the manipulator.
        hand_entity: Genesis entity for the gripper.
        mount: Mount transform between arm and gripper frames.
        arm_controller: Controller for manipulator DoFs.
        gripper_controller: Controller for gripper DoFs.
        layout: Flat action layout for arm and gripper commands.
    """

    def __init__(
        self,
        *,
        manipulator: ModularManipulatorModel,
        gripper: GripperModel,
        arm_control_options: ControllerOptions,
        gripper_control_options: ControllerOptions,
        scene: gs.Scene,
        base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        visualize_tcp: bool = False,
    ):
        """Initialize the object.

        Args:
        manipulator: Modular manipulator model instance.
        gripper: Gripper model instance to attach.
        arm_control_options: Configuration for the arm controller.
        gripper_control_options: Configuration for the gripper controller.
        scene: Genesis scene that owns the robot entities.
        base_pos: Robot base position `[x, y, z]` in world meters.
        quat: Robot base orientation `[w, x, y, z]` in world frame.
        visualize_tcp: Whether to visualize the tool center point. Defaults to False.
        """
        self.manip = manipulator
        self.gripper = gripper
        self.scene = scene
        self.device = get_device()

        self.arm_ent = scene.add_entity(
            get_robot_morph(manipulator, pos=base_pos, quat=quat),
            material=gs.materials.Rigid(needs_coup=True, gravity_compensation=1.0),
        )
        self._link_names = [link.name for link in self.arm_ent.links]
        self._eef_link_idx = self._link_names.index(self.manip.end_link)
        gripper_key = str(
            getattr(
                gripper,
                "_registry_name",
                getattr(gripper.__class__, "_registry_name", ""),
            ),
        )
        adapter_config = manipulator.adapters.get(gripper_key)
        canonical_gripper_key = str(getattr(gripper.__class__, "_registry_name", ""))
        if adapter_config is None and canonical_gripper_key != gripper_key:
            adapter_config = manipulator.adapters.get(canonical_gripper_key)
        if adapter_config is None:
            available_adapters = ", ".join(sorted(manipulator.adapters.keys()))
            raise ValueError(
                f"Adapter missing: manipulator '{manipulator.__class__._registry_name}' "
                f"has no mount for gripper '{gripper_key}'. "
                f"Available: [{available_adapters}]",
            )
        adapter_config = dict(adapter_config)
        parent_link = str(adapter_config.pop("parent_link", manipulator.end_link))
        mount = GripperMount(**adapter_config)
        mount_morph = create_mount_morph(mount)
        mount_ent = None

        if mount_morph:
            mount_ent = scene.add_entity(
                mount_morph,
                material=gs.materials.Rigid(needs_coup=True, gravity_compensation=1.0),
            )
            link_entities(
                self.arm_ent,
                mount_ent,
                parent_link_name=parent_link,
                child_link_name=mount_ent.links[0].name,
            )
            parent_link = mount_ent.links[0].name

        self.hand_ent = scene.add_entity(
            get_robot_morph(
                gripper,
                pos=mount.gripper_pos,
                quat=mount.gripper_quat,
            ),
            material=gs.materials.Rigid(
                needs_coup=True,
                gravity_compensation=1.0,
            ),
            visualize_contact=bool(getattr(gripper, "visualize_contact", False)),
        )

        link_entities(
            mount_ent or self.arm_ent,
            self.hand_ent,
            parent_link_name=parent_link,
            child_link_name=gripper.root_link,
        )
        self.arm_ctrl = make_ctrl(
            arm_control_options,
            entity=self.arm_ent,
            model=self.manip,
        )
        self.hand_ctrl = make_ctrl(
            gripper_control_options,
            entity=self.hand_ent,
            model=self.gripper,
        )
        self.dof = int(self.manip.dof) + int(self.gripper.dof)
        self.tactile: RigidPadTactile | None = None
        self._layout = None
        self._act_dim = None
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
        self.hand_ctrl.post_build()
        tactile_enabled = bool(getattr(self.gripper, "tactile", False))
        tactile_links = tuple(getattr(self.gripper, "tactile_links", ()) or ())
        if tactile_enabled and tactile_links:
            self.tactile = RigidPadTactile(
                self.hand_ent,
                link_names=tactile_links,
                grid_size=getattr(self.gripper, "tactile_grid_size", None),
                point_distance=float(
                    getattr(self.gripper, "tactile_point_distance", 0.002) or 0.002,
                ),
            )
        self._layout = build_single_layout(
            arm_ctrl=self.arm_ctrl,
            grip_ctrl=self.hand_ctrl,
        )
        self._act_dim = self._layout.total_dim

    def reset(self, env_idx=None) -> None:
        """Reset the arm and gripper to their home joint configurations.

        Args:
            env_idx: Optional indices of environments to reset. If None, resets all.
        """
        self.arm_ctrl.reset(env_idx=env_idx)
        self.hand_ctrl.reset(env_idx=env_idx)
        self.arm_ctrl.set_home_position(env_idx=env_idx)
        self.hand_ctrl.set_home_position(env_idx=env_idx)

    def get_obs(self) -> dict[str, Any]:
        """Retrieves the observation dictionary for the robot."""
        qpos_arm = self.arm_ent.get_dofs_position()
        qvel_arm = self.arm_ent.get_dofs_velocity()
        qtorque_arm = self.arm_ent.get_dofs_force()

        qpos_hand = self.hand_ent.get_dofs_position()
        qvel_hand = self.hand_ent.get_dofs_velocity()
        qtorque_hand = self.hand_ent.get_dofs_force()

        eef_pos, eef_quat = get_end_effector_pose(
            self.arm_ent,
            self._eef_link_idx,
            qpos=qpos_arm,
        )
        manipulator_observation = {
            "qpos": qpos_arm,
            "qvel": qvel_arm,
            "qtorque": qtorque_arm,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
        }
        tcp_pos, tcp_quat = get_tool_center_point_pose(self)
        gripper_observation = {
            "qpos": qpos_hand,
            "qvel": qvel_hand,
            "qtorque": qtorque_hand,
            "tcp_pos": tcp_pos,
            "tcp_quat": tcp_quat,
        }
        if self.tactile is not None:
            if bool(getattr(self, "tactile_summary_only", False)):
                gripper_observation["tactile"] = self.tactile.read_summary()
            else:
                gripper_observation["tactile"] = self.tactile.read()

        return {"manipulator": manipulator_observation, "gripper": gripper_observation}
