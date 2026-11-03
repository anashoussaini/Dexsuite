# tests/camera/test_05_disable_and_validation.py
import pytest

import dexsuite as ds
from dexsuite.core.camera import CameraSystem
from dexsuite.options import CamerasOptions, StaticCamOptions
from dexsuite.options.builders import normalize_make_options


def test_disable_all_cameras_with_none():
    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=None,  # explicit disable
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        assert obs["cameras"] == {}
    finally:
        env.close()


def test_camerasoptions_requires_valid_presets_for_static_strings():
    with pytest.raises(ValueError):
        _ = CamerasOptions(static={"bad": "does_not_exist"})


def test_camerasoptions_accepts_preset_static_and_dynamic_strings():
    # This is a smoke check against YAML lookups
    _ = CamerasOptions(static={"front": "front"}, dynamic={"wrist": "wrist_cam"})


def test_camera_gui_kw_validation_and_alias():
    assert normalize_make_options({"gui": True})["camera_gui"] is True

    with pytest.raises(TypeError, match="both 'gui' and 'camera_gui'"):
        normalize_make_options({"gui": True, "camera_gui": False})

    with pytest.raises(TypeError, match="camera_gui mapping values must be bools"):
        normalize_make_options({"camera_gui": {"front": 1}})


def test_camera_system_gui_mapping_uses_exact_camera_names():
    camera_system = object.__new__(CameraSystem)
    camera_system.gui = {"front": True}
    spec = StaticCamOptions(pos=(0, 0, 1), lookat=(0, 0, 0), gui=False)

    assert camera_system._camera_gui("front", spec) is True
    assert camera_system._camera_gui("wrist", spec) is False
