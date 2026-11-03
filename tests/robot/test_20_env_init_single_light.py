import pytest
import torch
from gymnasium import spaces

import dexsuite as ds


def zeros_like_space(space):
    if isinstance(space, spaces.Box):
        return torch.zeros(space.shape, dtype=torch.float32)
    raise TypeError(f"Unsupported action space type: {type(space)}")


@pytest.mark.slow
def test_env_init_single_modular_minimal():
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
        assert "state" in obs and isinstance(obs["state"], dict)
        assert env.action_space is not None

        act = zeros_like_space(env.action_space)  # <-- use shape-correct zeros
        env.step(act)
    finally:
        env.close()
