import unittest

import torch

from dexsuite.environments.base_env import BaseEnv


class _FakeScene:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0
        self.viewer = None

    def reset(self):
        self.reset_calls += 1

    def step(self):
        self.step_calls += 1


class _FakeRobot:
    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.reset_calls = []
        self.applied_actions = []

    def reset(self, env_idx=None):
        self.reset_calls.append(env_idx)

    def apply_action_validated(self, action):
        self.applied_actions.append(action.clone())

    def get_obs(self):
        return {
            "gripper": {
                "tcp_pos": torch.zeros((self.n_envs, 3), dtype=torch.float32),
            },
        }


class _FakeCameraSystem:
    def render_all(self):
        return {}


def _fake_env(
    *,
    n_envs: int = 1,
    act_dim: int = 3,
    horizon: int | None = None,
    success=None,
):
    env = object.__new__(BaseEnv)
    env.scene = _FakeScene()
    env.robot = _FakeRobot(n_envs)
    env.camera_sys = _FakeCameraSystem()
    env.device = torch.device("cpu")
    env.n_envs = n_envs
    env._act_dim = act_dim
    env.horizon = horizon
    env._n_phys_per_ctrl = 1
    env._built = True
    env._t = torch.zeros(n_envs, dtype=torch.int64)

    if n_envs == 1:
        env._current_action = torch.zeros(act_dim, dtype=torch.float32)
    else:
        env._current_action = torch.zeros((n_envs, act_dim), dtype=torch.float32)
    env._last_action = torch.zeros_like(env._current_action)

    def on_episode_start(envs_idx=None):
        del envs_idx

    def compute_reward(obs):
        del obs
        return torch.zeros(n_envs, dtype=torch.float32)

    def get_info(obs):
        del obs
        return {}

    def is_failure(obs):
        del obs
        return torch.zeros(n_envs, dtype=torch.bool)

    def is_success(obs):
        del obs
        if success is None:
            return torch.zeros(n_envs, dtype=torch.bool)
        return torch.as_tensor(success, dtype=torch.bool)

    env._on_episode_start = on_episode_start
    env._step = lambda: None
    env._get_extra_obs = dict
    env._compute_reward = compute_reward
    env._get_info = get_info
    env._is_failure = is_failure
    env._is_success = is_success
    return env


class TestBaseEnvActionState(unittest.TestCase):
    def test_reset_clears_current_and_last_action(self):
        env = _fake_env()
        env._current_action[:] = torch.tensor([1.0, 2.0, 3.0])
        env._last_action[:] = torch.tensor([4.0, 5.0, 6.0])

        obs, _ = BaseEnv.reset(env)

        self.assertTrue(torch.equal(env._current_action, torch.zeros(3)))
        self.assertTrue(torch.equal(env._last_action, torch.zeros(3)))
        self.assertTrue(torch.equal(obs["state"]["other"]["action"], torch.zeros(3)))
        self.assertTrue(
            torch.equal(obs["state"]["other"]["last_action"], torch.zeros(3)),
        )

    def test_reset_env_idx_clears_only_selected_action_rows(self):
        env = _fake_env(n_envs=2)
        env._current_action[:] = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        env._last_action[:] = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        env._t[:] = torch.tensor([5, 6])

        BaseEnv.reset_env_idx(env, torch.tensor([0]))

        self.assertTrue(torch.equal(env._current_action[0], torch.zeros(3)))
        self.assertTrue(torch.equal(env._last_action[0], torch.zeros(3)))
        self.assertTrue(
            torch.equal(env._current_action[1], torch.tensor([2.0, 2.0, 2.0])),
        )
        self.assertTrue(
            torch.equal(env._last_action[1], torch.tensor([4.0, 4.0, 4.0])),
        )
        self.assertEqual(env._t.tolist(), [0, 6])

    def test_step_auto_reset_does_not_restore_terminal_action_single_env(self):
        env = _fake_env(horizon=1)

        obs, _reward, _terminated, _truncated, _info = BaseEnv.step(
            env,
            torch.tensor([1.0, 2.0, 3.0]),
        )

        self.assertTrue(torch.equal(env._current_action, torch.zeros(3)))
        self.assertTrue(torch.equal(env._last_action, torch.zeros(3)))
        self.assertTrue(torch.equal(obs["state"]["other"]["action"], torch.zeros(3)))
        self.assertTrue(
            torch.equal(obs["state"]["other"]["last_action"], torch.zeros(3)),
        )

    def test_step_auto_reset_keeps_last_action_for_non_reset_envs(self):
        env = _fake_env(n_envs=2, success=[True, False])
        action = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        BaseEnv.step(env, action)

        self.assertTrue(torch.equal(env._current_action[0], torch.zeros(3)))
        self.assertTrue(torch.equal(env._last_action[0], torch.zeros(3)))
        self.assertTrue(
            torch.equal(env._current_action[1], torch.tensor([2.0, 2.0, 2.0])),
        )
        self.assertTrue(
            torch.equal(env._last_action[1], torch.tensor([2.0, 2.0, 2.0])),
        )
