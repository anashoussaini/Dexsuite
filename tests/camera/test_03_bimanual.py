# tests/camera/test_03_bimanual.py
from camera_helpers import (
    assert_modalities_exact,
    assert_pack_shapes,
)

import dexsuite as ds


def test_bimanual_wrist_expansion_rgb_only():
    """'wrist' expands to left/right wrists for bimanual.
    Default modalities ('rgb',) unless overridden.
    """
    env = ds.make(
        "bimanual_reach",
        manipulator=("franka", "franka"),
        gripper=("robotiq", "robotiq"),
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=["front", "wrist"],  # → left_wrist & right_wrist internally
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        cams = obs["cameras"]
        assert set(cams.keys()) == {"front", "left_wrist", "right_wrist"}
        for pack in cams.values():
            assert set(pack.keys()) == {"rgb"}
            assert_pack_shapes(pack, height=224, width=224)
    finally:
        env.close()


def test_bimanual_with_all_modalities():
    env = ds.make(
        "bimanual_reach",
        manipulator=("franka", "franka"),
        gripper=("robotiq", "robotiq"),
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=["front", "wrist"],
        modalities=("rgb", "depth", "segmentation", "normal"),
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        cams = obs["cameras"]
        assert set(cams.keys()) == {"front", "left_wrist", "right_wrist"}
        for pack in cams.values():
            assert_modalities_exact(pack, ("rgb", "depth", "segmentation", "normal"))
            assert_pack_shapes(pack, height=224, width=224)
    finally:
        env.close()
