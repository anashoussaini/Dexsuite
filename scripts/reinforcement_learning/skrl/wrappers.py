"""skrl wrapper for DexSuite reach-style tasks."""

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from skrl.envs.wrappers.torch.base import Wrapper


def _resolve_device(env: Any, device: torch.device | str | None = None) -> torch.device:
    """Return the torch device used by wrapper tensors."""
    if device is not None:
        return torch.device(device)
    env_device = getattr(env, "device", None)
    if env_device is not None:
        return torch.device(env_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DexsuiteReachWrapper(Wrapper):
    """DexSuite reach wrapper for skrl (torch).

    The wrapper flattens the nested DexSuite observation dict into a single
    1D vector per environment and adapts reset-by-index behavior to skrl.

    Observation vector layout (per environment):
        [tcp_pos, target_pos, manipulator.qpos, manipulator.qvel, other.action]

    Notes:
        - Actions are moved to self.device and cast to float32.
        - Input actions may be shaped (B, D) or (B, 1, D); the wrapper squeezes
          the middle dimension when present.
        - Rewards and done flags are returned as shape (B, 1).
    """

    def __init__(self, env: Any, device: torch.device | str | None = None) -> None:
        """Initialize the skrl wrapper around a DexSuite environment."""
        super().__init__(env)
        self._device = _resolve_device(env, device)

        # ---- Inspect nested observation spaces for dims ----
        obs_sp = self._env.observation_space
        state_sp = obs_sp["state"]

        tcp_sp = state_sp["gripper"]["tcp_pos"]
        tgt_sp = state_sp["other"]["target_pos"]
        qpos_sp = state_sp["manipulator"]["qpos"]
        qvel_sp = state_sp["manipulator"]["qvel"]

        # Try to locate 'other.action' under 'state' first, then top-level
        act_obs_sp = None
        try:
            act_obs_sp = state_sp["other"]["action"]
        except KeyError:
            try:
                act_obs_sp = obs_sp["other"]["action"]
            except KeyError as e:
                raise KeyError(
                    'Could not find observation space for ["other"]["action"] '
                    "(tried under ['state']['other'] and top-level ['other']).",
                ) from e

        self._tcp_dim = int(tcp_sp.shape[-1])
        self._tgt_dim = int(tgt_sp.shape[-1])
        self._qpos_dim = int(qpos_sp.shape[-1])
        self._qvel_dim = int(qvel_sp.shape[-1])
        self._actobs_dim = int(
            act_obs_sp.shape[-1],
        )  # dimension of obs["..."]["action"]

        self._obs_dim = (
            self._tcp_dim
            + self._tgt_dim
            + self._qpos_dim
            + self._qvel_dim
            + self._actobs_dim
        )

        # ---- Action space: ensure per-env Box(D,) ----
        act_sp = self._env.action_space
        if isinstance(act_sp, spaces.Box) and len(act_sp.shape) > 1:
            act_dim = int(act_sp.shape[-1])
            try:
                low = np.asarray(act_sp.low)
                high = np.asarray(act_sp.high)
                if low.shape == act_sp.shape and high.shape == act_sp.shape:
                    act_low = low.reshape(-1, act_dim)[0].astype(np.float32)
                    act_high = high.reshape(-1, act_dim)[0].astype(np.float32)
                else:
                    act_low = -np.ones(act_dim, dtype=np.float32)
                    act_high = np.ones(act_dim, dtype=np.float32)
            except (AttributeError, TypeError, ValueError):
                act_low = -np.ones(act_dim, dtype=np.float32)
                act_high = np.ones(act_dim, dtype=np.float32)
            self._act_space = spaces.Box(
                low=act_low,
                high=act_high,
                shape=(act_dim,),
                dtype=np.float32,
            )
        else:
            self._act_space = act_sp  # already per-env shape

        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space:
        """Return the flattened per-environment observation space."""
        return self._obs_space

    @property
    def action_space(self) -> gym.Space:
        """Return the per-environment action space expected by skrl."""
        return self._act_space

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments."""
        return int(getattr(self._env, "n_envs", 1))

    @property
    def num_agents(self) -> int:
        """Return the number of agents, fixed to one for DexSuite tasks."""
        return 1

    @property
    def device(self) -> torch.device:
        """Return the torch device used by the wrapper tensors."""
        return self._device

    @staticmethod
    def _squeeze_batch_1d(x: torch.Tensor) -> torch.Tensor:
        """Squeeze a singleton middle batch dimension when present."""
        if x.ndim == 3 and x.shape[1] == 1:
            return x.squeeze(1)
        return x

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        """Promote a 1D observation tensor to shape `(1, D)`."""
        if x.ndim == 1:
            return x.unsqueeze(0)
        return x

    @staticmethod
    def _to_batch_column(
        x: Any,
        batch_size: int,
        *,
        dtype,
        device,
    ) -> torch.Tensor:
        """Convert scalar or vector data to a minimal `(B, 1)` tensor."""
        t = torch.as_tensor(x, dtype=dtype, device=device)
        if t.ndim == 2:
            return t if t.shape[1] == 1 else t[:, :1]
        if t.ndim == 1:
            return t.reshape(batch_size, 1)
        if t.ndim == 0:
            return t.view(1, 1).expand(batch_size, 1)
        return t.reshape(batch_size, -1)[:, :1]

    @staticmethod
    def _path_exists(obs: Any, path: Sequence[str]) -> bool:
        """Return whether a nested mapping path exists."""
        node = obs
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return False
            node = node[key]
        return True

    def _grab_as_tensor(
        self,
        obs: Any,
        path: Sequence[str],
        *,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Read a nested observation path and return it as a 2D torch tensor."""
        node = obs
        for k in path:
            node = node[k]
        t = node if torch.is_tensor(node) else torch.as_tensor(node)
        return self._ensure_2d(self._squeeze_batch_1d(t)).to(
            self.device,
            dtype=dtype,
        )

    def _grab_first_as_tensor(
        self,
        obs: Any,
        path_options: Sequence[Sequence[str]],
        *,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Try multiple observation paths and return the first valid tensor."""
        for path in path_options:
            if self._path_exists(obs, path):
                return self._grab_as_tensor(obs, path, dtype=dtype)
        raise KeyError(
            f"None of the candidate paths exist: {path_options}",
        )

    def _extract_parts(
        self,
        obs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and batch-align the observation components used by skrl."""
        tcp = self._grab_as_tensor(obs, ["state", "gripper", "tcp_pos"])
        tgt = self._grab_as_tensor(obs, ["state", "other", "target_pos"])
        qpos = self._grab_as_tensor(obs, ["state", "manipulator", "qpos"])
        qvel = self._grab_as_tensor(obs, ["state", "manipulator", "qvel"])
        act_obs = self._grab_first_as_tensor(
            obs,
            (["state", "other", "action"], ["other", "action"]),
        )

        # Align batch sizes (expand 1 to B as needed).
        batch_size = max(
            tcp.shape[0],
            tgt.shape[0],
            qpos.shape[0],
            qvel.shape[0],
            act_obs.shape[0],
        )

        def _align(x: torch.Tensor) -> torch.Tensor:
            return (
                x
                if x.shape[0] == batch_size
                else (x.expand(batch_size, x.shape[1]) if x.shape[0] == 1 else x)
            )

        return _align(tcp), _align(tgt), _align(qpos), _align(qvel), _align(act_obs)

    def _flatten_obs(self, obs: Any) -> torch.Tensor:
        """Flatten a DexSuite reach observation into a `(B, D)` tensor."""
        tcp, tgt, qpos, qvel, act_obs = self._extract_parts(obs)
        return torch.cat([tcp, tgt, qpos, qvel, act_obs], dim=-1)

    def _env_uses_autoreset(self, info: Any) -> bool:
        """Return whether DexSuite already reset done rows inside step."""
        return isinstance(info, dict) and "needs_reset" in info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Step the wrapped environment and return skrl-shaped tensors."""
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actions = self._squeeze_batch_1d(actions)
        actions = actions.to(self.device, dtype=torch.float32)

        obs, reward, terminated, truncated, info = self._env.step(actions * 0.5)

        observation = self._flatten_obs(obs)
        batch_size = observation.shape[0]

        reward2d = self._to_batch_column(
            reward,
            batch_size,
            dtype=torch.float32,
            device=self._device,
        )
        terminated2d = self._to_batch_column(
            terminated,
            batch_size,
            dtype=torch.bool,
            device=self._device,
        )
        truncated2d = self._to_batch_column(
            truncated,
            batch_size,
            dtype=torch.bool,
            device=self._device,
        )

        # Some gym-like envs require the wrapper to reset done rows. BaseEnv
        # already auto-resets and marks this through info["needs_reset"], so do
        # not reset those rows a second time.
        done_mask = terminated2d[:, 0] | truncated2d[:, 0]  # (B,)
        if (
            done_mask.any().item()
            and hasattr(self._env, "reset_env_idx")
            and not self._env_uses_autoreset(info)
        ):
            reset_idx = done_mask.nonzero(as_tuple=False).squeeze(-1)  # cuda idx
            reset_idx_cpu = reset_idx.detach().long().cpu()

            reset_obs, _ = self._env.reset_env_idx(reset_idx_cpu)

            # Extract new parts after reset
            new_rows = self._flatten_obs(reset_obs)

            # Patch: handle whether reset_obs returned B rows or only len(reset_idx)
            if new_rows.shape[0] == observation.shape[0]:
                observation[reset_idx] = new_rows[reset_idx]
            elif new_rows.shape[0] == reset_idx.numel():
                observation[reset_idx] = new_rows
            else:
                observation[reset_idx] = new_rows[: reset_idx.numel()]

        return observation, reward2d, terminated2d, truncated2d, info

    def reset(self) -> tuple[torch.Tensor, Any]:
        """Reset the wrapped environment and return a flattened observation."""
        obs, info = self._env.reset()

        return self._flatten_obs(obs), info

    def reset_by_idx(
        self,
        env_idx: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Reset selected vectorized env rows and return flattened observations."""
        if not hasattr(self._env, "reset_env_idx"):
            return self.reset()
        reset_idx = None
        if env_idx is not None:
            reset_idx = torch.as_tensor(env_idx, dtype=torch.long).detach().cpu()
        obs, info = self._env.reset_env_idx(reset_idx)
        return self._flatten_obs(obs), info

    def render(self, *args, **kwargs) -> Any:
        """Forward rendering to the wrapped environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()


class DexsuiteLiftWrapper(Wrapper):
    """Parallel-friendly DexSuite lift wrapper for skrl."""

    _ARM_ACTION_SCALE_RAD = 2.0
    _ARM_MAX_DELTA_RAD = 0.15

    def __init__(self, env: Any, device: torch.device | str | None = None) -> None:
        """Initialize the skrl wrapper around a DexSuite lift environment."""
        super().__init__(env)
        self._device = _resolve_device(env, device)
        self._arm_dim = int(getattr(self._env.robot.arm_ctrl, "dof", 7))
        self._last_arm_q_target: torch.Tensor | None = None
        self._arm_home_q, self._arm_lo, self._arm_hi = self._arm_joint_limits()

        state_space = self._env.observation_space["state"]
        tcp_space = state_space["gripper"]["tcp_pos"]
        cube_space = state_space["other"]["cube_pos"]
        target_space = state_space["other"]["target_pos"]
        qpos_space = state_space["manipulator"]["qpos"]
        qvel_space = state_space["manipulator"]["qvel"]
        gripper_space = state_space["gripper"]
        try:
            action_obs_space = state_space["other"]["action"]
        except KeyError:
            action_obs_space = self._env.observation_space["other"]["action"]
        self._tactile_pad_names = ()
        try:
            tactile_space = gripper_space["tactile"]
        except (KeyError, TypeError):
            tactile_space = None
        self._tactile_pad_names = self._tactile_pad_names_from_space(tactile_space)
        tactile_dim = 2 * len(self._tactile_pad_names)

        self._obs_dim = int(
            tcp_space.shape[-1]
            + cube_space.shape[-1]
            + target_space.shape[-1]
            + tcp_space.shape[-1]
            + cube_space.shape[-1]
            + qpos_space.shape[-1]
            + qvel_space.shape[-1]
            + action_obs_space.shape[-1]
            + tactile_dim,
        )

        action_space = self._env.action_space
        action_dim = int(action_space.shape[-1])
        self._act_space = spaces.Box(
            low=np.ones(action_dim, dtype=np.float32) * action_space.low.min(),
            high=np.ones(action_dim, dtype=np.float32) * action_space.high.max(),
            shape=(action_dim,),
            dtype=np.float32,
        )
        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @staticmethod
    def _tactile_pad_names_from_space(tactile_space: Any) -> tuple[str, ...]:
        if isinstance(tactile_space, spaces.Dict):
            tactile_space = tactile_space.spaces
        pad_names = [
            name
            for name in ("left_elastomer", "right_elastomer")
            if DexsuiteLiftWrapper._has_tactile_pad(tactile_space, name)
        ]
        return tuple(pad_names) if len(pad_names) == 2 else ()

    @staticmethod
    def _has_tactile_pad(tactile_space: Any, name: str) -> bool:
        try:
            return name in tactile_space
        except TypeError:
            pass
        try:
            tactile_space[name]
        except (KeyError, TypeError):
            return False
        return True

    @property
    def observation_space(self) -> spaces.Box:
        """Return the flattened per-environment observation space."""
        return self._obs_space

    @property
    def action_space(self) -> spaces.Box:
        """Return the per-environment action space."""
        return self._act_space

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments."""
        return int(getattr(self._env, "n_envs", 1))

    @property
    def num_agents(self) -> int:
        """Return the number of agents, fixed to one."""
        return 1

    @property
    def device(self) -> torch.device:
        """Return the torch device used by the wrapper."""
        return self._device

    def _grab(self, obs: Any, path: Sequence[str]) -> torch.Tensor:
        """Read a nested observation path and return a 2D float32 tensor."""
        value = obs
        for part in path:
            value = value[part]
        tensor = torch.as_tensor(value, device=self._device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _arm_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return arm home and joint limits used by Isaac-like action scaling."""
        ctrl = self._env.robot.arm_ctrl
        home_q = getattr(ctrl, "home_q", None)
        lo = getattr(ctrl, "_lo_1d", None)
        hi = getattr(ctrl, "_hi_1d", None)
        if home_q is None or lo is None or hi is None:
            raise RuntimeError(
                "DexsuiteLiftWrapper requires a built normalized joint_position "
                "arm controller with home_q and joint limits.",
            )
        home_q = torch.as_tensor(home_q, device=self._device, dtype=torch.float32)
        lo = torch.as_tensor(lo, device=self._device, dtype=torch.float32)
        hi = torch.as_tensor(hi, device=self._device, dtype=torch.float32)
        return home_q[: self._arm_dim], lo[: self._arm_dim], hi[: self._arm_dim]

    def _normalise_arm_q(self, q_target: torch.Tensor) -> torch.Tensor:
        """Map arm joint targets in radians to normalized controller targets."""
        denom = (self._arm_hi - self._arm_lo).clamp(min=1e-6)
        return 2.0 * (q_target - self._arm_lo) / denom - 1.0

    def _current_arm_q_from_obs(self, obs: Any) -> torch.Tensor:
        """Read current arm qpos from an observation dict."""
        return self._grab(obs, ["state", "manipulator", "qpos"])[:, : self._arm_dim]

    def _set_last_arm_target_from_obs(
        self,
        obs: Any,
        env_idx: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Initialize or patch the slew-limit reference from observed qpos."""
        arm_q = self._current_arm_q_from_obs(obs).detach().clone()
        if env_idx is None or self._last_arm_q_target is None:
            self._last_arm_q_target = arm_q
            return
        idx = torch.as_tensor(env_idx, dtype=torch.long, device=self._device)
        if arm_q.shape[0] == self.num_envs:
            self._last_arm_q_target[idx] = arm_q[idx]
        else:
            self._last_arm_q_target[idx] = arm_q[: idx.numel()]

    def _map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map policy actions to conservative normalized joint-position targets."""
        mapped = actions.clone()
        arm_policy = mapped[:, : self._arm_dim].clamp(-1.0, 1.0)
        q_target = self._arm_home_q + self._ARM_ACTION_SCALE_RAD * arm_policy
        q_target = torch.minimum(torch.maximum(q_target, self._arm_lo), self._arm_hi)

        if self._last_arm_q_target is None:
            self._last_arm_q_target = self._arm_home_q.view(1, -1).expand(
                actions.shape[0],
                -1,
            ).clone()
        if self._last_arm_q_target.shape[0] != actions.shape[0]:
            self._last_arm_q_target = self._last_arm_q_target[:1].expand(
                actions.shape[0],
                -1,
            ).clone()

        delta = (q_target - self._last_arm_q_target).clamp(
            -self._ARM_MAX_DELTA_RAD,
            self._ARM_MAX_DELTA_RAD,
        )
        q_target = self._last_arm_q_target + delta
        self._last_arm_q_target = q_target.detach().clone()
        mapped[:, : self._arm_dim] = self._normalise_arm_q(q_target).clamp(-1.0, 1.0)
        if mapped.shape[1] >= self._arm_dim + 2:
            gripper_raw = actions[:, self._arm_dim : self._arm_dim + 2].clamp(
                -1.0,
                1.0,
            )
            close_cmd = (gripper_raw[:, 1] - gripper_raw[:, 0]).mul(0.5).clamp(
                -1.0,
                1.0,
            )
            mapped[:, self._arm_dim] = -close_cmd
            mapped[:, self._arm_dim + 1] = close_cmd
        return mapped

    def _extract_parts(
        self,
        obs: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Extract the observation components used by the wrapper."""
        tcp = self._grab(obs, ["state", "gripper", "tcp_pos"])
        cube = self._grab(obs, ["state", "other", "cube_pos"])
        target = self._grab(obs, ["state", "other", "target_pos"])
        qpos = self._grab(obs, ["state", "manipulator", "qpos"])
        qvel = self._grab(obs, ["state", "manipulator", "qvel"])
        action_obs = self._grab(obs, ["state", "other", "action"])
        tactile = self._tactile_summary(obs, tcp.shape[0])
        tcp_to_grasp = cube + torch.tensor(
            [0.0, 0.0, 0.018],
            dtype=torch.float32,
            device=self._device,
        ) - tcp
        cube_to_target = target - cube
        return tcp, cube, target, tcp_to_grasp, cube_to_target, qpos, qvel, action_obs, tactile

    def _tactile_summary(self, obs: Any, batch_size: int) -> torch.Tensor:
        """Return compact tactile features `[contacts..., force_norms...]`."""
        if not self._tactile_pad_names:
            return torch.empty((batch_size, 0), dtype=torch.float32, device=self._device)
        tactile = obs["state"]["gripper"].get("tactile", {})
        features = []
        for pad_name in self._tactile_pad_names:
            pad = tactile.get(pad_name, {})
            contact = torch.as_tensor(
                pad.get("contact", torch.zeros(batch_size, 1)),
                device=self._device,
                dtype=torch.float32,
            ).reshape(batch_size, -1)[:, :1]
            features.append(contact)
        for pad_name in self._tactile_pad_names:
            pad = tactile.get(pad_name, {})
            force_norm = torch.as_tensor(
                pad.get("force_norm", torch.zeros(batch_size, 1)),
                device=self._device,
                dtype=torch.float32,
            ).reshape(batch_size, -1)[:, :1]
            features.append(force_norm)
        return torch.cat(features, dim=-1)

    def _flatten_obs(self, obs: Any) -> torch.Tensor:
        """Flatten a DexSuite lift observation into a `(B, D)` tensor."""
        tcp, cube, target, tcp_to_grasp, cube_to_target, qpos, qvel, action_obs, tactile = (
            self._extract_parts(obs)
        )
        return torch.cat(
            [tcp, cube, target, tcp_to_grasp, cube_to_target, qpos, qvel, action_obs, tactile],
            dim=-1,
        )

    @staticmethod
    def _env_uses_autoreset(info: Any) -> bool:
        return isinstance(info, dict) and "needs_reset" in info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Step the wrapped environment and return skrl-shaped outputs."""
        actions = torch.as_tensor(actions, device=self._device, dtype=torch.float32)
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        obs, reward, terminated, truncated, info = self._env.step(self._map_actions(actions))

        observation = self._flatten_obs(obs)

        batch_size = observation.shape[0]
        reward = torch.as_tensor(
            reward,
            device=self._device,
            dtype=torch.float32,
        ).reshape(batch_size, 1)
        terminated = torch.as_tensor(
            terminated,
            device=self._device,
            dtype=torch.bool,
        ).reshape(batch_size, 1)
        truncated = torch.as_tensor(
            truncated,
            device=self._device,
            dtype=torch.bool,
        ).reshape(batch_size, 1)

        done_mask = terminated[:, 0] | truncated[:, 0]
        if self._env_uses_autoreset(info):
            reset_idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if reset_idx.numel() > 0:
                self._set_last_arm_target_from_obs(obs, reset_idx)
        if (
            done_mask.any()
            and hasattr(self._env, "reset_env_idx")
            and not self._env_uses_autoreset(info)
        ):
            reset_idx = done_mask.nonzero(as_tuple=False).squeeze(-1).cpu()
            reset_obs, _ = self._env.reset_env_idx(reset_idx)
            new_rows = self._flatten_obs(reset_obs)
            observation[reset_idx] = new_rows[: len(reset_idx)]
            self._set_last_arm_target_from_obs(reset_obs, reset_idx)

        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs) -> Any:
        """Forward rendering to the wrapped environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()

    def reset(self) -> tuple[torch.Tensor, Any]:
        """Reset the wrapped environment and return a flattened observation."""
        obs, info = self._env.reset()
        self._set_last_arm_target_from_obs(obs)
        return self._flatten_obs(obs), info

    def reset_by_idx(
        self,
        env_idx: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Reset selected vectorized env rows and return flattened observations."""
        if not hasattr(self._env, "reset_env_idx"):
            return self.reset()
        reset_idx = None
        if env_idx is not None:
            reset_idx = torch.as_tensor(env_idx, dtype=torch.long).detach().cpu()
        obs, info = self._env.reset_env_idx(reset_idx)
        self._set_last_arm_target_from_obs(obs, reset_idx)
        return self._flatten_obs(obs), info
