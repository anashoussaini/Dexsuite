# tests/camera/test_02_direct_names_and_modalities.py
import pytest
from camera_helpers import (
    assert_modalities_exact,
    assert_pack_shapes,
)

import dexsuite as ds


def test_direct_names_and_all_modalities():
    """Use direct camera names + global modalities applied to all."""
    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=["front", "overhead", "wrist"],
        modalities=("rgb", "depth", "segmentation", "normal"),
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        cams = obs["cameras"]
        assert set(cams.keys()) == {"front", "overhead", "wrist"}
        for pack in cams.values():
            assert_modalities_exact(pack, ("rgb", "depth", "segmentation", "normal"))
            assert_pack_shapes(pack, height=224, width=224)
    finally:
        env.close()


def test_direct_invalid_camera_name_raises():
    with pytest.raises(ValueError):
        ds.make(
            "stack",
            manipulator="franka",
            gripper="robotiq",
            arm_control="osc_pose",
            gripper_control="joint_position",
            cameras=["no_such_cam"],  # must exist in YAML
        )


def test_direct_modalities_missing_rgb_raises():
    with pytest.raises(ValueError):
        ds.make(
            "stack",
            manipulator="franka",
            gripper="robotiq",
            arm_control="osc_pose",
            gripper_control="joint_position",
            cameras=["front"],
            modalities=("depth",),  # rgb must be present
        )


def test_direct_modalities_invalid_entry_raises():
    with pytest.raises(ValueError):
        ds.make(
            "stack",
            manipulator="franka",
            gripper="robotiq",
            arm_control="osc_pose",
            gripper_control="joint_position",
            cameras=["front"],
            modalities=("rgb", "albedo"),  # not allowed
        )
