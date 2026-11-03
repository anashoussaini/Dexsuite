"""Evaluate PPO checkpoints on centralized DexSuite SKRL environments."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

try:
    from scripts.reinforcement_learning.skrl.configs import env_choices
    from scripts.reinforcement_learning.skrl.envs import make_skrl_env
    from scripts.reinforcement_learning.skrl.models import PPOPolicy, PPOValue
    from scripts.reinforcement_learning.skrl.utils import (
        as_action_tensor,
        infer_policy_action_dim_from_checkpoint,
        map_actions_tensor,
        resolve_checkpoint_path,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.reinforcement_learning"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import env_choices
    from envs import make_skrl_env
    from models import PPOPolicy, PPOValue
    from utils import (
        as_action_tensor,
        infer_policy_action_dim_from_checkpoint,
        map_actions_tensor,
        resolve_checkpoint_path,
    )


# ===========================
# Evaluation
# ===========================


@torch.no_grad()
def evaluate(  # noqa: PLR0915
    checkpoint: str,
    env_name: str = "reach",
    n_envs: int = 64,
    episodes: int = 256,
    render_mode: str | None = None,
    robot_preset: str | None = None,
    action_map: str = "auto",  # "auto" | "trim_last" | "pad_zero_last" | "none"
    seed: int | None = None,
) -> tuple[float, int, int]:
    """Run evaluation and return (success_rate, successes, total_episodes)."""
    if seed is not None:
        torch.manual_seed(seed)

    ckpt_path = resolve_checkpoint_path(checkpoint)
    ckpt_actions = infer_policy_action_dim_from_checkpoint(ckpt_path)

    env = make_skrl_env(
        env_name,
        n_envs=n_envs,
        render_mode=render_mode,
        robot_preset=robot_preset,
    )
    obs, _ = env.reset()
    device = env.device

    env_actions = int(env.action_space.shape[-1])

    # ---- Agent action space (prefer to match env; map only if needed) ----
    agent_action_dim = (
        ckpt_actions
        if (ckpt_actions is not None and ckpt_actions != env_actions)
        else env_actions
    )
    agent_action_space = spaces.Box(
        low=np.full((agent_action_dim,), -1.0, dtype=np.float32),
        high=np.full((agent_action_dim,), 1.0, dtype=np.float32),
        shape=(agent_action_dim,),
        dtype=np.float32,
    )

    # Models & agent config (preprocessors as in training)
    models = {
        "policy": PPOPolicy(
            env.observation_space,
            agent_action_space,
            device,
            clip_actions=True,
        ),
        "value": PPOValue(env.observation_space, agent_action_space, device),
    }
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["mixed_precision"] = True  # harmless at eval, matches train environment
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    agent = PPO(
        models=models,
        memory=None,  # no memory needed for eval
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=agent_action_space,
        device=device,
    )
    agent.init()

    print(f"[eval] Loading checkpoint: {ckpt_path}")
    agent.load(ckpt_path)
    agent.set_mode("eval")
    agent.set_running_mode("eval")

    num_envs = int(getattr(env, "num_envs", getattr(env, "n_envs", 1)))
    episodes_done = torch.zeros(num_envs, dtype=torch.int64, device=obs.device)
    successes = torch.zeros(num_envs, dtype=torch.int64, device=obs.device)

    total_steps = 0
    target_episodes = int(episodes)

    try:
        while int(episodes_done.sum().item()) < target_episodes:
            states_t = obs  # already torch on cuda:0
            actions_out = agent.act(states_t, timestep=total_steps, timesteps=10**12)

            actions_t = as_action_tensor(actions_out, device=obs.device)

            # Map actions if agent and env dims differ
            if agent_action_dim != env_actions:
                actions_t = map_actions_tensor(
                    actions_t,
                    action_map,
                    agent_action_dim,
                    env_actions,
                )

            obs, _reward, terminated, truncated, info = env.step(actions_t)

            # dones
            term_t = (
                terminated
                if isinstance(terminated, torch.Tensor)
                else torch.as_tensor(terminated, device=obs.device)
            )
            trunc_t = (
                truncated
                if isinstance(truncated, torch.Tensor)
                else torch.as_tensor(truncated, device=obs.device)
            )
            done = term_t.view(num_envs, -1)[:, 0].to(torch.bool) | trunc_t.view(
                num_envs,
                -1,
            )[:, 0].to(torch.bool)

            # successes from info (env should populate a per-env bool)
            success_mask = info.get(
                "success",
                torch.zeros(num_envs, dtype=torch.bool, device=obs.device),
            )
            if not isinstance(success_mask, torch.Tensor):
                success_mask = torch.as_tensor(
                    success_mask,
                    dtype=torch.bool,
                    device=obs.device,
                )

            successes += (success_mask & done).to(torch.int64)
            episodes_done += done.to(torch.int64)

            total_steps += 1
    finally:
        env.close()

    total_eps = int(episodes_done.sum().item())
    total_succ = int(successes.sum().item())
    rate = total_succ / max(1, total_eps)

    print(
        f"\n[eval] Episodes: {total_eps}  |  Successes: {total_succ}  |  Success rate: {rate * 100:.2f}%",
    )
    return rate, total_succ, total_eps


def main() -> None:
    """Run PPO checkpoint evaluation from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint on Dexsuite Reach (parallel).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt or experiment dir with checkpoints/",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=256,
        help="Total episodes to evaluate (across all envs)",
    )
    parser.add_argument("--env", dest="env_name", choices=env_choices(), default="reach")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Render mode",
    )

    parser.add_argument(
        "--robot-preset",
        choices=[
            "franka_with_gripper_normalized",
            "legacy_franka_robotiq",
            "franka_umi_tactile_joint_position",
            "franka_umi_tactile_abs",
        ],
        default=None,
    )

    # If checkpoint/env action dims differ:
    parser.add_argument(
        "--action_map",
        type=str,
        default="auto",
        choices=["auto", "trim_last", "pad_zero_last", "none"],
        help="How to adapt checkpoint actions to env dim if they differ",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        env_name=args.env_name,
        n_envs=args.n_envs,
        episodes=args.episodes,
        render_mode=args.render_mode,
        robot_preset=args.robot_preset,
        action_map=args.action_map,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
