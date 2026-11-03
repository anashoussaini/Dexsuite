from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

pytest.importorskip("skrl")

from scripts.reinforcement_learning.skrl.configs import (
    ENV_CONFIGS,
    algo_choices,
    env_choices,
    validate_env_algo,
)
from scripts.reinforcement_learning.skrl.train import build_parser
from scripts.reinforcement_learning.skrl.wrappers import DexsuiteReachWrapper


class _FakeReachEnv:
    def __init__(
        self,
        *,
        n_envs: int = 3,
        act_dim: int = 4,
        auto_reset: bool = True,
    ) -> None:
        self.device = torch.device("cpu")
        self.n_envs = n_envs
        self.act_dim = act_dim
        self.auto_reset = auto_reset
        self.reset_calls = 0
        self.reset_env_idx_calls: list[torch.Tensor | None] = []
        self.step_actions: list[torch.Tensor] = []
        self.action = torch.zeros((n_envs, act_dim), dtype=torch.float32)
        self.reset_marker = torch.zeros(n_envs, dtype=torch.float32)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "gripper": spaces.Dict(
                            {
                                "tcp_pos": spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    shape=(3,),
                                    dtype=np.float32,
                                ),
                            },
                        ),
                        "other": spaces.Dict(
                            {
                                "target_pos": spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    shape=(3,),
                                    dtype=np.float32,
                                ),
                                "action": spaces.Box(
                                    -1.0,
                                    1.0,
                                    shape=(act_dim,),
                                    dtype=np.float32,
                                ),
                            },
                        ),
                        "manipulator": spaces.Dict(
                            {
                                "qpos": spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                                "qvel": spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                            },
                        ),
                    },
                ),
                "cameras": spaces.Dict({}),
            },
        )
        self.action_space = spaces.Box(
            low=-np.ones((n_envs, act_dim), dtype=np.float32),
            high=np.ones((n_envs, act_dim), dtype=np.float32),
            shape=(n_envs, act_dim),
            dtype=np.float32,
        )

    def _obs(self) -> dict:
        row = torch.arange(self.n_envs, dtype=torch.float32).unsqueeze(-1)
        return {
            "state": {
                "gripper": {"tcp_pos": torch.cat([row, row + 0.1, row + 0.2], dim=1)},
                "other": {
                    "target_pos": torch.stack(
                        [
                            self.reset_marker,
                            self.reset_marker + 1.0,
                            self.reset_marker + 2.0,
                        ],
                        dim=1,
                    ),
                    "action": self.action.clone(),
                },
                "manipulator": {
                    "qpos": torch.cat([row + 2.0, row + 3.0], dim=1),
                    "qvel": torch.cat([row + 4.0, row + 5.0], dim=1),
                },
            },
            "cameras": {},
        }

    def reset(self):
        self.reset_calls += 1
        self.action.zero_()
        return self._obs(), {"reset": True}

    def reset_env_idx(self, env_idx):
        idx = None
        if env_idx is not None:
            idx = torch.as_tensor(env_idx, dtype=torch.long).detach().cpu()
            self.action[idx] = 0.0
            self.reset_marker[idx] += 100.0
        else:
            self.action.zero_()
            self.reset_marker += 100.0
        self.reset_env_idx_calls.append(idx)
        return self._obs(), {"reset_env_idx": idx}

    def step(self, actions):
        self.step_actions.append(actions.detach().cpu().clone())
        self.action.copy_(actions.detach().cpu())

        terminated = torch.tensor([False, True, False], dtype=torch.bool)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool)
        info = {"success": terminated.clone()}
        if self.auto_reset:
            reset_mask = terminated | truncated
            info["needs_reset"] = reset_mask
            self.action[reset_mask] = 0.0
            self.reset_marker[reset_mask] += 10.0

        reward = torch.arange(self.n_envs, dtype=torch.float32) + 0.25
        return self._obs(), reward, terminated, truncated, info

    def render(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


def test_reach_wrapper_uses_env_device_and_shapes_parallel_outputs():
    env = _FakeReachEnv()
    wrapper = DexsuiteReachWrapper(env)

    obs, info = wrapper.reset()

    assert wrapper.device == torch.device("cpu")
    assert wrapper.num_envs == 3
    assert wrapper.action_space.shape == (4,)
    assert wrapper.observation_space.shape == (14,)
    assert obs.shape == (3, 14)
    assert obs.device.type == "cpu"
    assert info == {"reset": True}


def test_reach_wrapper_accepts_skrl_action_shape_and_does_not_double_reset():
    env = _FakeReachEnv(auto_reset=True)
    wrapper = DexsuiteReachWrapper(env)
    actions = torch.ones((3, 1, 4), dtype=torch.float32)

    obs, reward, terminated, truncated, info = wrapper.step(actions)

    assert env.step_actions[0].shape == (3, 4)
    assert torch.allclose(env.step_actions[0], torch.full((3, 4), 0.5))
    assert env.reset_env_idx_calls == []
    assert obs.shape == (3, 14)
    assert reward.shape == (3, 1)
    assert terminated.shape == (3, 1)
    assert truncated.shape == (3, 1)
    assert reward.dtype == torch.float32
    assert terminated.dtype == torch.bool
    assert truncated.dtype == torch.bool
    assert torch.equal(info["needs_reset"], torch.tensor([False, True, False]))
    assert obs[1, 3].item() == pytest.approx(10.0)


def test_reach_wrapper_resets_done_rows_when_env_does_not_autoreset():
    env = _FakeReachEnv(auto_reset=False)
    wrapper = DexsuiteReachWrapper(env)

    obs, _reward, terminated, _truncated, _info = wrapper.step(torch.ones((3, 4)))

    assert torch.equal(terminated[:, 0], torch.tensor([False, True, False]))
    assert len(env.reset_env_idx_calls) == 1
    assert torch.equal(env.reset_env_idx_calls[0], torch.tensor([1]))
    assert obs[1, 3].item() == pytest.approx(100.0)


def test_reach_wrapper_reset_by_idx_delegates_to_env_and_flattens_obs():
    env = _FakeReachEnv()
    wrapper = DexsuiteReachWrapper(env)

    obs, info = wrapper.reset_by_idx(torch.tensor([0, 2], device=wrapper.device))

    assert obs.shape == (3, 14)
    assert torch.equal(env.reset_env_idx_calls[0], torch.tensor([0, 2]))
    assert torch.equal(info["reset_env_idx"], torch.tensor([0, 2]))
    assert obs[0, 3].item() == pytest.approx(100.0)
    assert obs[1, 3].item() == pytest.approx(0.0)
    assert obs[2, 3].item() == pytest.approx(100.0)


def test_skrl_registry_declares_reach_algorithms():
    assert "reach" in env_choices()
    assert "lift" in env_choices()
    assert {"ppo", "td3"}.issubset(set(algo_choices()))
    assert ENV_CONFIGS["reach"].supported_algorithms == ("ppo", "td3")
    assert ENV_CONFIGS["lift"].supported_algorithms == ("ppo",)
    validate_env_algo("reach", "ppo")
    validate_env_algo("reach", "td3")
    validate_env_algo("lift", "ppo")


def test_unified_train_parser_accepts_algo_env_and_legacy_spellings():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--algo",
            "td3",
            "--env",
            "reach",
            "--num_steps",
            "2",
            "--n_envs",
            "2",
        ],
    )

    assert args.algo == "td3"
    assert args.env_name == "reach"
    assert args.num_steps == 2
    assert args.n_envs == 2
