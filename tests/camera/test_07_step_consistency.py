# tests/camera/test_07_step_consistency.py
import torch
from camera_helpers import assert_modalities_exact

import dexsuite as ds


def test_camera_pack_names_and_modalities_persist_across_steps():
    env = ds.make(
        "stack",
        manipulator="franka",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=["front", "wrist"],
        modalities=("rgb", "depth"),
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        names0 = set(obs["cameras"].keys())
        for p in obs["cameras"].values():
            assert_modalities_exact(p, ("rgb", "depth"))

        action = torch.zeros(
            env.action_space.shape,
            dtype=torch.float32,
            device=env.device,
        )
        for _ in range(3):
            obs, *_ = env.step(action)
            names = set(obs["cameras"].keys())
            assert names == names0
            for pack in obs["cameras"].values():
                assert_modalities_exact(pack, ("rgb", "depth"))
    finally:
        env.close()
