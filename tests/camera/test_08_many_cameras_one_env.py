# tests/camera/test_08_many_cameras_one_env.py
from __future__ import annotations

import math

import pytest
import torch

import dexsuite as ds
from dexsuite.options import CamerasOptions, StaticCamOptions


def _supports_framebuffers(env) -> bool:
    """Return True if the active Genesis offscreen backend supports FBOs.

    Some backends (notably OSMesa) don't support framebuffer objects; Genesis'
    offscreen wrapper may then fall back to a fixed 32x32 viewport, which makes
    per-camera resolution assertions impossible.
    """
    try:
        rast = env.scene._visualizer._rasterizer
        if not getattr(rast, "_offscreen", True):  # viewer path
            return True
        renderer = rast._renderer
        platform = renderer._platform
        return bool(platform.supports_framebuffers())
    except Exception:
        # If Genesis internals change, keep the test conservative.
        return False


def _expected_hw_from_res_wh(res_wh: tuple[int, int]) -> tuple[int, int]:
    """Genesis camera uses res=(W,H) but returns arrays shaped (H,W[,C])."""
    w, h = int(res_wh[0]), int(res_wh[1])
    return h, w


@pytest.fixture(scope="module")
def env_many_cameras():
    # 18 static cameras + 2 bimanual wrists = 20 total camera packs.
    static: dict[str, StaticCamOptions | str] = {"front": "front"}
    lookat = (0.0, 0.0, 0.35)

    n_custom = 17
    radius = 1.25
    base_z = 0.75
    for i in range(n_custom):
        theta = 2.0 * math.pi * (i / n_custom)
        name = f"cam{i + 1:02d}"
        pos = (
            radius * math.cos(theta),
            radius * math.sin(theta),
            base_z + 0.01 * i,
        )
        # Use non-square resolutions; res is (W,H).
        res = (96 + 4 * i, 80 + 2 * i)
        static[name] = StaticCamOptions(pos=pos, lookat=lookat, fov=70.0, res=res)

    cams = CamerasOptions(
        static=static,
        dynamic={"wrist": "wrist_cam"},
        modalities=("rgb", "depth", "segmentation", "normal"),
    )

    env = ds.make(
        "stack",
        manipulator=("franka", "franka"),
        gripper=("robotiq", "robotiq"),
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=cams,
        render_mode="rgb_array",
    )
    try:
        yield env
    finally:
        env.close()


def test_many_cameras_single_build_shapes_modalities_and_naming(env_many_cameras):
    env = env_many_cameras
    obs, _ = env.reset()

    packs = obs["cameras"]
    assert isinstance(packs, dict)

    # Naming: bimanual wrist expansion should be left/right prefixed, no suffix-duplication.
    static_names = set(env.camera_sys.cam_options.static.keys())
    expected_names = static_names | {"left_wrist", "right_wrist"}
    assert set(packs.keys()) == expected_names
    assert all(
        not n.endswith("_left_left") and not n.endswith("_right_right")
        for n in packs.keys()
    )

    enabled_modalities = tuple(env.camera_sys.cam_options.modalities)
    assert enabled_modalities == ("rgb", "depth", "segmentation", "normal")

    # Basic type + dtype checks (independent of resolution backend).
    for cam_name, pack in packs.items():
        assert set(pack.keys()) == set(enabled_modalities), cam_name

        rgb = pack["rgb"]
        assert torch.is_tensor(rgb)
        assert rgb.ndim == 3 and rgb.shape[-1] == 3
        assert rgb.dtype == torch.uint8

        depth = pack["depth"]
        assert torch.is_tensor(depth)
        assert depth.ndim == 2
        assert depth.dtype == torch.float32

        seg = pack["segmentation"]
        assert torch.is_tensor(seg)
        assert seg.ndim == 2
        assert seg.dtype in (torch.int32, torch.int64, torch.uint8)

        normal = pack["normal"]
        assert torch.is_tensor(normal)
        assert normal.ndim == 3 and normal.shape[-1] == 3
        assert normal.dtype in (torch.float32, torch.uint8)

    # Resolution assertions only when the backend supports per-camera res.
    if _supports_framebuffers(env):
        # Expected resolution is derived from the config objects.
        expected_res_wh: dict[str, tuple[int, int]] = {}
        for name, spec in env.camera_sys.cam_options.static.items():
            expected_res_wh[name] = tuple(spec.res)
        for name, (_cam, spec, _ent, _root) in env.camera_sys._dyn_specs.items():
            expected_res_wh[name] = tuple(spec.res)

        # Assert observation_space and returned tensors match the requested res.
        cams_space = env.observation_space.spaces["cameras"]
        for cam_name, pack in packs.items():
            exp_h, exp_w = _expected_hw_from_res_wh(expected_res_wh[cam_name])

            assert pack["rgb"].shape[:2] == (exp_h, exp_w), cam_name
            assert pack["depth"].shape == (exp_h, exp_w), cam_name
            assert pack["segmentation"].shape == (exp_h, exp_w), cam_name
            assert pack["normal"].shape[:2] == (exp_h, exp_w), cam_name

            cam_space = cams_space.spaces[cam_name]
            assert cam_space.spaces["rgb"].shape == (exp_h, exp_w, 3), cam_name
            assert cam_space.spaces["depth"].shape == (exp_h, exp_w), cam_name
            assert cam_space.spaces["segmentation"].shape == (exp_h, exp_w), cam_name
            assert cam_space.spaces["normal"].shape == (exp_h, exp_w, 3), cam_name

    # Step consistency: camera names and modalities must persist across steps.
    action = torch.zeros_like(env._current_action)
    for _ in range(2):
        obs, *_ = env.step(action)
        assert set(obs["cameras"].keys()) == expected_names
        for pack in obs["cameras"].values():
            assert set(pack.keys()) == set(enabled_modalities)
