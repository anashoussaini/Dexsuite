# tests/camera/test_04_dataclass_custom.py
import pytest
from camera_helpers import (
    assert_modalities_exact,
    assert_pack_shapes,
)

import dexsuite as ds
from dexsuite.options import CamerasOptions, StaticCamOptions


def test_dataclass_presets_plus_custom_static_res_and_modalities():
    """Mix a preset static camera with a custom StaticCamOptions to test res/shape.
    Use global modalities from CamerasOptions (dataclass).
    """
    # Choose a non-square resolution to verify HxW mapping (Genesis returns HxW)
    custom_res = (200, 300)  # H=200, W=300 expected in arrays
    cams = CamerasOptions(
        static={
            "front": "front",  # preset
            "custom_overhead": StaticCamOptions(
                pos=(0.0, 0.0, 2.0),
                lookat=(0.0, 0.0, 0.6),
                fov=50.0,
                res=(custom_res[1], custom_res[0]),  # Genesis commonly takes (W,H)
            ),
        },
        dynamic={"wrist": "wrist_cam"},
        modalities=("rgb", "depth", "segmentation", "normal"),
    )

    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=cams,
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        packs = obs["cameras"]

        # Cameras that must exist
        assert "front" in packs and "custom_overhead" in packs and "wrist" in packs

        # Modalities exact across all cams
        for p in packs.values():
            assert_modalities_exact(p, ("rgb", "depth", "segmentation", "normal"))

        # Shapes:
        #  - front & wrist from YAML: 224 x 224
        assert_pack_shapes(packs["front"], height=224, width=224)
        assert_pack_shapes(packs["wrist"], height=224, width=224)
        #  - custom_overhead uses the overridden res; array is HxW
        assert_pack_shapes(
            packs["custom_overhead"],
            height=custom_res[0],
            width=custom_res[1],
        )
    finally:
        env.close()


def test_dataclass_modalities_precedence_over_kw():
    """Passing both cameras=CamerasOptions(...) and modalities=... must raise."""
    cams = CamerasOptions(
        static={"front": "front"},
        dynamic={"wrist": "wrist_cam"},
        modalities=("rgb", "depth", "normal"),
    )
    with pytest.raises(
        ValueError,
        match="When passing cameras=CamerasOptions\\(\\.\\.\\.\\), set modalities on that object",
    ):
        ds.make(
            "stack",
            manipulator="franka",
            gripper="robotiq",
            arm_control="osc_pose",
            gripper_control="joint_position",
            cameras=cams,
            modalities=("rgb",),
            render_mode="rgb_array",
        )
