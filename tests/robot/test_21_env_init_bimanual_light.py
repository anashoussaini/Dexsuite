# tests/robot/test_21_env_init_bimanual_light.py
import pytest
import torch

import dexsuite as ds


@pytest.mark.slow
def test_env_init_bimanual_modular_minimal():
    env = ds.make(
        "bimanual_reach",
        manipulator=("franka", "franka"),
        gripper=("robotiq", "robotiq"),
        arm_control="osc_pose",
        gripper_control="joint_position",
        render_mode="rgb_array",
    )
    try:
        obs, _ = env.reset()
        assert "state" in obs
        assert env.action_space is not None
        act = torch.zeros(env.action_space.shape, dtype=torch.float32)
        env.step(act)
    finally:
        env.close()
