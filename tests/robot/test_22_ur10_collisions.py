import pytest
import torch

import dexsuite as ds


@pytest.mark.slow
def test_ur10_shoulder_collision_clears_table_and_allows_y_motion():
    env = ds.make(
        "stack",
        manipulator="ur10",
        gripper="robotiq",
        arm_control="osc_pose",
        gripper_control="joint_position",
        cameras=None,
        render_mode=None,
    )
    try:
        env.reset()

        contacts = env.robot.arm_ent.get_contacts()
        assert contacts["penetration"].numel() == 0

        tcp_before = env.robot.get_obs()["gripper"]["tcp_pos"].detach().clone()
        action = torch.zeros(
            env.action_space.shape, dtype=torch.float32, device=env.device
        )
        action[1] = 1.0
        action[-1] = -1.0

        for _ in range(5):
            env.step(action)

        tcp_after = env.robot.get_obs()["gripper"]["tcp_pos"].detach().clone()
        delta_y = (tcp_after - tcp_before).reshape(-1)[1]
        assert delta_y > 0.01

        contacts = env.robot.arm_ent.get_contacts()
        assert contacts["penetration"].numel() == 0
    finally:
        env.close()
