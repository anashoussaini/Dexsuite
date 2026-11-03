from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from gymnasium import spaces

import dexsuite as ds
from dexsuite.options import ArmOptions, ControllerOptions, LayoutOptions, RobotOptions
from dexsuite.utils.options_utils import load_layout_preset

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
GRIP_CTRL_BAD = ("osc_pose", "osc_pose_abs", "ik_pose")


def _zeros_like_space(space):
    if not isinstance(space, spaces.Box):
        raise TypeError(f"Expected spaces.Box action space, got {type(space)}")
    return torch.zeros(space.shape, dtype=torch.float32)


def _assert_unit_box_bounds(space):
    if not isinstance(space, spaces.Box):
        raise TypeError(f"Expected spaces.Box action space, got {type(space)}")
    assert np.allclose(space.low, -1.0), f"low bounds not -1: {space.low}"
    assert np.allclose(space.high, 1.0), f"high bounds not 1: {space.high}"


@pytest.mark.slow
def test_bimanual_minimal_env_default_layout_side_by_side():
    """Bimanual: franka+franka.
    If no preset is provided, RobotOptions should default to a sensible bimanual preset
    (we require 'side_by_side') and resolve left/right poses from YAML.
    """
    rob = RobotOptions(
        type_of_robot="bimanual",
        left=ArmOptions(manipulator="franka", gripper="robotiq"),
        right=ArmOptions(manipulator="franka", gripper="robotiq"),
        layout=LayoutOptions(),  # no preset given → side_by_side default expected
    )

    # Spec: layout must be resolved to poses and preset should be set to 'side_by_side'
    assert rob.layout.left is not None and rob.layout.right is not None
    assert isinstance(rob.layout.preset, str) and rob.layout.preset == "side_by_side"

    # Compare against layouts/franka.yaml side_by_side blocks.
    fr = load_layout_preset("franka")
    expected_left = fr["side_by_side"]["left"]
    expected_right = fr["side_by_side"]["right"]
    assert np.allclose(rob.layout.left.pos, np.asarray(expected_left["pos"], dtype=float))
    assert np.allclose(
        rob.layout.right.pos,
        np.asarray(expected_right["pos"], dtype=float),
    )

    env = ds.make("bimanual_reach", robot=rob, cameras=None, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(env.action_space, spaces.Box)

        # Bounds normalized
        _assert_unit_box_bounds(env.action_space)

        # Flat torch action
        env.step(_zeros_like_space(env.action_space))
    finally:
        env.close()


def _yaw_to_rad(block: dict) -> float:
    """Helper to mirror Pose normalization (yaw_deg → yaw_rad, default 0)."""
    if block.get("quat") is not None:
        return 0.0
    if block.get("yaw_rad") is not None:
        return float(block["yaw_rad"])
    if block.get("yaw_deg") is not None:
        return float(block["yaw_deg"]) * math.pi / 180.0
    return 0.0


def test_bimanual_layout_resolution_mixed_models_options_only():
    """Layout resolution is per-model:
      - 'ur5' resolves from layouts/ur5.yaml.
      - 'franka' resolves from layouts/franka.yaml.
    This is an OPTIONS-LEVEL check (no env creation) so we don't depend on a UR5 registry.
    """
    rob = RobotOptions(
        type_of_robot="bimanual",
        left=ArmOptions(manipulator="ur5", gripper="robotiq"),
        right=ArmOptions(manipulator="franka", gripper="robotiq"),
        layout=LayoutOptions(preset="side_by_side"),
    )

    # Expected left from ur5, right from franka.
    ur5 = load_layout_preset("ur5")
    franka = load_layout_preset("franka")
    left_expected = ur5["side_by_side"]["left"]
    right_expected = franka["side_by_side"]["right"]

    assert np.allclose(rob.layout.left.pos, np.asarray(left_expected["pos"], dtype=float))
    assert np.allclose(
        rob.layout.right.pos,
        np.asarray(right_expected["pos"], dtype=float),
    )
    assert math.isclose(
        rob.layout.left.yaw_rad,
        _yaw_to_rad(left_expected),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )
    assert math.isclose(
        rob.layout.right.yaw_rad,
        _yaw_to_rad(right_expected),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )


@pytest.mark.parametrize("bad_grip", GRIP_CTRL_BAD)
def test_bimanual_gripper_rejects_non_joint_controllers(bad_grip):
    rob = RobotOptions(
        type_of_robot="bimanual",
        left=ArmOptions(
            manipulator="franka",
            gripper="robotiq",
            manipulator_controller=ControllerOptions(name="osc_pose"),
            gripper_controller=ControllerOptions(name=bad_grip),
        ),
        right=ArmOptions(
            manipulator="franka",
            gripper="robotiq",
            manipulator_controller=ControllerOptions(name="osc_pose"),
            gripper_controller=ControllerOptions(name=bad_grip),
        ),
        layout=LayoutOptions(preset="side_by_side"),
    )
    with pytest.raises((ValueError, KeyError, TypeError)):
        ds.make("bimanual_reach", robot=rob, cameras=None, render_mode="rgb_array")


def test_bimanual_gripper_pair_length_mismatch_raises():
    """If someone supplies a bad gripper tuple size, robot builder should error.
    (This one intentionally exercises the direct API, still relevant for callers.)
    """
    with pytest.raises(ValueError):
        ds.make(
            "bimanual_reach",
            manipulator=("franka", "franka"),
            gripper=("robotiq",),  # wrong length
            cameras=None,
            render_mode="rgb_array",
        )


def test_bimanual_unknown_manipulator_errors_at_build():
    rob = RobotOptions(
        type_of_robot="bimanual",
        left=ArmOptions(manipulator="franka", gripper="robotiq"),
        right=ArmOptions(manipulator="unknown_model", gripper="robotiq"),
        layout=LayoutOptions(preset="side_by_side"),
    )
    # Accept a broader set of exceptions so test is robust to registry guard location.
    with pytest.raises((ValueError, KeyError, RuntimeError, TypeError)):
        ds.make("bimanual_reach", robot=rob, cameras=None, render_mode="rgb_array")


@pytest.mark.slow
def test_bimanual_env_cameras_none_and_step():
    """Minimal bimanual bring-up with cameras disabled and a safe zero step."""
    rob = RobotOptions(
        type_of_robot="bimanual",
        left=ArmOptions(manipulator="franka", gripper="robotiq"),
        right=ArmOptions(manipulator="franka", gripper="robotiq"),
        layout=LayoutOptions(preset="side_by_side"),
    )
    env = ds.make("bimanual_reach", robot=rob, cameras=None, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        if "cameras" in obs:
            assert isinstance(obs["cameras"], dict)
            assert len(obs["cameras"]) == 0
        _assert_unit_box_bounds(env.action_space)
        env.step(_zeros_like_space(env.action_space))
    finally:
        env.close()
