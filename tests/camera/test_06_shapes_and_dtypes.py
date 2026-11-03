# tests/camera/test_06_shapes_and_dtypes.py
from camera_helpers import (
    assert_modalities_exact,
    assert_pack_shapes,
    is_float_dtype,
    is_integer_dtype,
    to_numpy,
)

import dexsuite as ds


def test_all_modalities_shapes_and_dtypes():
    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=["front", "wrist"],
        modalities=("rgb", "depth", "segmentation", "normal"),
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        packs = obs["cameras"]
        assert set(packs.keys()) == {"front", "wrist"}
        for pack in packs.values():
            assert_modalities_exact(pack, ("rgb", "depth", "segmentation", "normal"))
            assert_pack_shapes(pack, height=224, width=224)

            # Extra lightweight dtype checks:
            rgb = to_numpy(pack["rgb"])
            assert rgb.dtype == rgb.astype("uint8").dtype
            dep = to_numpy(pack["depth"])
            assert is_float_dtype(dep)
            seg = to_numpy(pack["segmentation"])
            assert is_integer_dtype(seg)
            nrm = to_numpy(pack["normal"])
            assert nrm.ndim == 3 and nrm.shape[-1] == 3
    finally:
        env.close()
