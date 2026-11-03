from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

import dexsuite as ds
from dexsuite.options import ArmOptions, ControllerOptions, RobotOptions

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ARM_CTRL_ALL = (
    "joint_position",
    "joint_velocity",
    "joint_torque",
    "osc_pose",
    "osc_pose_abs",
    "ik_pose",
)
GRIP_CTRL_JOINT = ("joint_position", "joint_velocity", "joint_torque")
GRIP_CTRL_BAD = ("osc_pose", "osc_pose_abs", "ik_pose")  # gripper must not use these


def _zeros_like_space(space):
    """Create a torch zero action matching the env action space."""
    if not isinstance(space, spaces.Box):
        raise TypeError(f"Expected spaces.Box action space, got {type(space)}")
    return torch.zeros(space.shape, dtype=torch.float32)


def _assert_unit_box_bounds(space):
    if not isinstance(space, spaces.Box):
        raise TypeError(f"Expected spaces.Box action space, got {type(space)}")
    assert np.allclose(space.low, -1.0), f"low bounds not -1: {space.low}"
    assert np.allclose(space.high, 1.0), f"high bounds not 1: {space.high}"


@pytest.mark.slow
def test_single_minimal_modular_env_init():
    """Single: franka+robotiq; default controllers from YAML; cameras disabled."""
    rob = RobotOptions(
        type_of_robot="single",
        single=ArmOptions(manipulator="franka", gripper="robotiq"),
        # layout for single is explicit pose if needed; we keep default (origin)
    )
    env = ds.make("stack", robot=rob, cameras=None, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs
        # cameras disabled → empty or missing cameras dict is OK (depends on your BaseEnv)
        if "cameras" in obs:
            assert isinstance(obs["cameras"], dict)
            assert len(obs["cameras"]) == 0

        # Flat torch action API
        assert isinstance(env.action_space, spaces.Box)

        # Controllers from YAML are normalized=True by default → [-1, 1] bounds
        _assert_unit_box_bounds(env.action_space)

        # Safe step: send flat torch zeros aligned with the action_space
        act = _zeros_like_space(env.action_space)
        env.step(act)
    finally:
        env.close()


@pytest.mark.parametrize("arm_name", ARM_CTRL_ALL)
@pytest.mark.parametrize("grip_name", GRIP_CTRL_JOINT)
@pytest.mark.slow
def test_single_controller_matrix_builds(arm_name, grip_name):
    """Single: sweep valid manipulator×gripper controller names.
    We expect env to build, reset, and step once with normalized [-1,1] bounds.
    """
    rob = RobotOptions(
        type_of_robot="single",
        single=ArmOptions(
            manipulator="franka",
            gripper="robotiq",
            manipulator_controller=ControllerOptions(name=arm_name),
            gripper_controller=ControllerOptions(name=grip_name),
        ),
    )
    env = ds.make("stack", robot=rob, cameras=None, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(env.action_space, spaces.Box)

        # Bounds should be normalized boxes throughout
        _assert_unit_box_bounds(env.action_space)

        # Step zeros
        env.step(_zeros_like_space(env.action_space))
    finally:
        env.close()


@pytest.mark.parametrize("bad_grip", GRIP_CTRL_BAD)
def test_single_gripper_rejects_non_joint_controllers(bad_grip):
    """Gripper must use a joint-based controller. Using OSC/IK controllers for a gripper should error.
    We accept a ValueError/KeyError/TypeError depending on where the check fires.
    """
    rob = RobotOptions(
        type_of_robot="single",
        single=ArmOptions(
            manipulator="franka",
            gripper="robotiq",
            manipulator_controller=ControllerOptions(name="osc_pose"),
            gripper_controller=ControllerOptions(name=bad_grip),
        ),
    )
    with pytest.raises((ValueError, KeyError, TypeError)):
        ds.make("stack", robot=rob, cameras=None, render_mode="rgb_array")


def test_single_unknown_manipulator_errors_at_build():
    rob = RobotOptions(
        type_of_robot="single",
        single=ArmOptions(manipulator="does_not_exist", gripper="robotiq"),
    )
    # Accept TypeError too in case the registry returns None and the factory calls it.
    with pytest.raises((ValueError, KeyError, RuntimeError, TypeError)):
        ds.make("stack", robot=rob, cameras=None, render_mode="rgb_array")
