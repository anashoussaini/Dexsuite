"""Training-time metric monitoring for SKRL environments."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import torch


class EpisodeStatsMonitor:
    """Track reward, success rate, and episode length for vectorized envs."""

    def __init__(
        self,
        env: Any,
        *,
        log_interval: int = 100,
        window_size: int = 100,
        print_stats: bool = True,
    ) -> None:
        self._env = env
        self._log_interval = max(int(log_interval), 1)
        self._print_stats = bool(print_stats)
        self._episode_returns = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_lengths = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._recent_returns: deque[float] = deque(maxlen=int(window_size))
        self._recent_successes: deque[float] = deque(maxlen=int(window_size))
        self._recent_lengths: deque[float] = deque(maxlen=int(window_size))
        self._total_episodes = 0
        self._total_successes = 0
        self._step_count = 0
        self._last_print_time = time.perf_counter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def num_envs(self) -> int:
        return int(getattr(self._env, "num_envs", getattr(self._env, "n_envs", 1)))

    @property
    def num_agents(self) -> int:
        return int(getattr(self._env, "num_agents", 1))

    @property
    def device(self) -> torch.device:
        return self._env.device

    def reset(self) -> tuple[torch.Tensor, Any]:
        obs, info = self._env.reset()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        return obs, info

    def reset_by_idx(
        self,
        env_idx: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        obs, info = self._env.reset_by_idx(env_idx)
        if env_idx is None:
            self._episode_returns.zero_()
            self._episode_lengths.zero_()
        else:
            idx = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
            self._episode_returns[idx] = 0.0
            self._episode_lengths[idx] = 0.0
        return obs, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        obs, reward, terminated, truncated, info = self._env.step(actions)
        self._step_count += 1

        reward_1d = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        ).reshape(self.num_envs, -1)[:, 0]
        terminated_1d = torch.as_tensor(
            terminated,
            dtype=torch.bool,
            device=self.device,
        ).reshape(self.num_envs, -1)[:, 0]
        truncated_1d = torch.as_tensor(
            truncated,
            dtype=torch.bool,
            device=self.device,
        ).reshape(self.num_envs, -1)[:, 0]
        done = terminated_1d | truncated_1d

        self._episode_returns += reward_1d
        self._episode_lengths += 1.0

        success = info.get(
            "success",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        success_1d = torch.as_tensor(
            success,
            dtype=torch.bool,
            device=self.device,
        ).reshape(self.num_envs)

        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            for idx_t in done_idx:
                idx = int(idx_t.item())
                ep_return = float(self._episode_returns[idx].detach().cpu().item())
                ep_length = float(self._episode_lengths[idx].detach().cpu().item())
                ep_success = float(success_1d[idx].detach().cpu().item())
                self._recent_returns.append(ep_return)
                self._recent_lengths.append(ep_length)
                self._recent_successes.append(ep_success)
                self._total_episodes += 1
                self._total_successes += int(ep_success)
            self._episode_returns[done_idx] = 0.0
            self._episode_lengths[done_idx] = 0.0

        self._attach_episode_metrics(info, reward_1d)
        if self._print_stats and self._step_count % self._log_interval == 0:
            self._print_line(reward_1d)
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs) -> Any:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    def _attach_episode_metrics(
        self,
        info: Any,
        reward_1d: torch.Tensor,
    ) -> None:
        if not isinstance(info, dict):
            return
        episode = info.setdefault("episode", {})
        episode["reward_step_mean"] = reward_1d.mean().detach()
        episode["reward_live_mean"] = self._episode_returns.mean().detach()
        episode["completed"] = torch.tensor(
            self._total_episodes,
            dtype=torch.float32,
            device=self.device,
        )
        episode["success_rate_total"] = torch.tensor(
            self._total_successes / max(self._total_episodes, 1),
            dtype=torch.float32,
            device=self.device,
        )
        if self._recent_returns:
            episode["reward_episode_mean"] = torch.tensor(
                sum(self._recent_returns) / len(self._recent_returns),
                dtype=torch.float32,
                device=self.device,
            )
            episode["success_rate"] = torch.tensor(
                sum(self._recent_successes) / len(self._recent_successes),
                dtype=torch.float32,
                device=self.device,
            )
            episode["episode_length_mean"] = torch.tensor(
                sum(self._recent_lengths) / len(self._recent_lengths),
                dtype=torch.float32,
                device=self.device,
            )

    def _print_line(self, reward_1d: torch.Tensor) -> None:
        now = time.perf_counter()
        dt = max(now - self._last_print_time, 1e-9)
        self._last_print_time = now
        env_steps_per_s = self.num_envs * self._log_interval / dt
        step_reward = float(reward_1d.mean().detach().cpu().item())
        live_return = float(self._episode_returns.mean().detach().cpu().item())
        if self._recent_returns:
            ep_reward = sum(self._recent_returns) / len(self._recent_returns)
            success_rate = sum(self._recent_successes) / len(self._recent_successes)
            ep_length = sum(self._recent_lengths) / len(self._recent_lengths)
            print(
                "[train] "
                f"step={self._step_count} "
                f"env_steps={self._step_count * self.num_envs} "
                f"reward_step={step_reward:.3f} "
                f"reward_ep={ep_reward:.3f} "
                f"success_rate={success_rate:.3f} "
                f"ep_len={ep_length:.1f} "
                f"episodes={self._total_episodes} "
                f"env_steps/s={env_steps_per_s:.1f}",
                flush=True,
            )
            return
        print(
            "[train] "
            f"step={self._step_count} "
            f"env_steps={self._step_count * self.num_envs} "
            f"reward_step={step_reward:.3f} "
            f"live_return={live_return:.3f} "
            "success_rate=n/a "
            "episodes=0 "
            f"env_steps/s={env_steps_per_s:.1f}",
            flush=True,
        )
