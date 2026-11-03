from __future__ import annotations

import pytest
import torch

from dexsuite.environments.single.baselines.lift import LiftEnv


class _PoseEntity:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.pos = torch.tensor(pos, dtype=torch.float32)
        self.quat = torch.tensor(quat, dtype=torch.float32)

    def get_pos(self) -> torch.Tensor:
        return self.pos.clone()

    def get_quat(self) -> torch.Tensor:
        return self.quat.clone()


def _lift_env() -> LiftEnv:
    env = object.__new__(LiftEnv)
    env.n_envs = 1
    env.device = torch.device("cpu")
    return env


def _obs(
    *,
    tcp=(0.4, 0.0, 0.25),
    cube=(0.0, 0.0, 0.025),
    target=(0.0, 0.0, 0.25),
    tactile: bool = False,
    left_force: float = 1.0,
    right_force: float = 1.0,
    action: torch.Tensor | None = None,
) -> dict:
    action_tensor = (
        torch.zeros((1, 9), dtype=torch.float32)
        if action is None
        else action.to(dtype=torch.float32).reshape(1, 9)
    )
    obs = {
        "state": {
            "gripper": {
                "tcp_pos": torch.tensor([tcp], dtype=torch.float32),
            },
            "other": {
                "cube_pos": torch.tensor([cube], dtype=torch.float32),
                "target_pos": torch.tensor([target], dtype=torch.float32),
                "action": action_tensor,
                "last_action": torch.zeros((1, 9), dtype=torch.float32),
            },
            "manipulator": {
                "qvel": torch.zeros((1, 7), dtype=torch.float32),
            },
        },
    }
    if tactile:
        obs["state"]["gripper"]["tactile"] = {
            "left_elastomer": {
                "contact": torch.ones((1, 1), dtype=torch.float32),
                "force_norm": torch.tensor([[left_force]], dtype=torch.float32),
            },
            "right_elastomer": {
                "contact": torch.ones((1, 1), dtype=torch.float32),
                "force_norm": torch.tensor([[right_force]], dtype=torch.float32),
            },
        }
    return obs


def test_lift_reward_progresses_through_lift_stages():
    env = _lift_env()

    start = env._compute_reward(_obs()).item()
    approach_contact = env._compute_reward(
        _obs(tcp=(0.0, 0.0, 0.045)),
    ).item()
    lifted = env._compute_reward(
        _obs(tcp=(0.0, 0.0, 0.14), cube=(0.0, 0.0, 0.12)),
    ).item()
    success = env._compute_reward(
        _obs(tcp=(0.0, 0.0, 0.235), cube=(0.0, 0.0, 0.22)),
    ).item()

    assert approach_contact > start
    assert lifted > approach_contact
    assert success > lifted


def test_tactile_observations_do_not_change_lift_reward():
    env = _lift_env()

    no_touch_reward = env._compute_reward(
        _obs(tcp=(0.0, 0.0, 0.045), tactile=False),
    ).item()
    touch_reward = env._compute_reward(
        _obs(tcp=(0.0, 0.0, 0.045), tactile=True, left_force=1.0, right_force=1.0),
    ).item()

    assert touch_reward == pytest.approx(no_touch_reward)


def test_lift_reward_keeps_single_env_tactile_shape_batched():
    env = _lift_env()
    obs = _obs(tcp=(0.0, 0.0, 0.045), tactile=True)
    obs["state"]["gripper"]["tcp_pos"] = obs["state"]["gripper"]["tcp_pos"][0]
    obs["state"]["other"]["cube_pos"] = obs["state"]["other"]["cube_pos"][0]
    obs["state"]["other"]["target_pos"] = obs["state"]["other"]["target_pos"][0]
    obs["state"]["other"]["action"] = obs["state"]["other"]["action"][0]
    obs["state"]["other"]["last_action"] = obs["state"]["other"]["last_action"][0]
    obs["state"]["manipulator"]["qvel"] = obs["state"]["manipulator"]["qvel"][0]

    reward = env._compute_reward(obs)

    assert reward.shape == (1,)


def test_lift_reward_uses_isaac_style_goal_terms_after_lift():
    env = _lift_env()

    table_terms = env._lift_reward_terms(_obs(tcp=(0.0, 0.0, 0.045)))
    lifted_terms = env._lift_reward_terms(
        _obs(tcp=(0.0, 0.0, 0.14), cube=(0.0, 0.0, 0.12)),
    )

    assert table_terms["lifted_score"].item() == pytest.approx(0.0)
    assert table_terms["goal_score"].item() == pytest.approx(0.0)
    assert lifted_terms["lifted_score"].item() == pytest.approx(1.0)
    assert lifted_terms["goal_score"].item() > 0.0


def test_action_rate_and_joint_velocity_penalties_reduce_lift_reward():
    env = _lift_env()
    base = _obs(tcp=(0.0, 0.0, 0.045))
    penalized = _obs(tcp=(0.0, 0.0, 0.045))
    penalized["state"]["other"]["action"] = torch.ones((1, 9), dtype=torch.float32)
    penalized["state"]["manipulator"]["qvel"] = torch.ones((1, 7), dtype=torch.float32)

    assert env._compute_reward(penalized).item() < env._compute_reward(base).item()


def test_close_action_is_rewarded_only_near_cube():
    env = _lift_env()
    close_action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])

    far = env._lift_reward_terms(_obs(action=close_action))["near_close_score"].item()
    near = env._lift_reward_terms(
        _obs(tcp=(0.0, 0.0, 0.045), action=close_action),
    )["near_close_score"].item()

    assert near > far


def test_lift_extra_obs_exposes_task_state_keys():
    env = _lift_env()
    env.cube = _PoseEntity((0.10, 0.20, 0.025))
    env.goal = _PoseEntity((0.10, 0.20, 0.25))

    extra = env._get_extra_obs()

    assert set(extra) == {"cube_pos", "cube_quat", "target_pos"}
