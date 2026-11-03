# tests/camera/test_01_defaults.py
from camera_helpers import (
    assert_pack_shapes,
)

import dexsuite as ds


def test_single_default_cameras_rgb_only():
    """Omit cameras so defaults kick in: ['front','wrist'] with modalities ('rgb',)."""
    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        cams = obs["cameras"]
        assert set(cams.keys()) == {"front", "wrist"}
        for name, pack in cams.items():
            assert set(pack.keys()) == {"rgb"}
            # Your YAML lists 224x224; enforce shape & dtype
            assert_pack_shapes(pack, height=224, width=224)
    finally:
        env.close()
