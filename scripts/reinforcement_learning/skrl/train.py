"""Unified SKRL training entrypoint for DexSuite tasks."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

try:
    from scripts.reinforcement_learning.skrl.configs import (
        ALGO_CONFIGS,
        ENV_CONFIGS,
        algo_choices,
        env_choices,
        validate_env_algo,
    )
    from scripts.reinforcement_learning.skrl.envs import make_skrl_env
    from scripts.reinforcement_learning.skrl.monitor import EpisodeStatsMonitor
    from scripts.reinforcement_learning.skrl.models import (
        PPOPolicy,
        PPOValue,
        TD3Policy,
        TD3Value,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.reinforcement_learning"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from configs import (
        ALGO_CONFIGS,
        ENV_CONFIGS,
        algo_choices,
        env_choices,
        validate_env_algo,
    )
    from envs import make_skrl_env
    from monitor import EpisodeStatsMonitor
    from models import PPOPolicy, PPOValue, TD3Policy, TD3Value


def _build_ppo(env, *, device, n_envs: int, experiment_dir: str):
    """Build a PPO agent for an already wrapped SKRL env."""
    rollouts_per_env = 256
    memory = RandomMemory(
        memory_size=rollouts_per_env,
        num_envs=n_envs,
        device=device,
        replacement=False,
    )
    models = {
        "policy": PPOPolicy(
            env.observation_space,
            env.action_space,
            device,
            clip_actions=True,
        ),
        "value": PPOValue(env.observation_space, env.action_space, device),
    }

    cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    cfg["rollouts"] = rollouts_per_env
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["rewards_shaper_scale"] = 1.0
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"] = 1.0
    cfg["mixed_precision"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = ALGO_CONFIGS["ppo"].write_interval
    cfg["experiment"]["checkpoint_interval"] = ALGO_CONFIGS[
        "ppo"
    ].checkpoint_interval
    cfg["experiment"]["directory"] = experiment_dir

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


def _build_td3(env, *, device, n_envs: int, experiment_dir: str):
    """Build a TD3 agent for an already wrapped SKRL env."""
    memory = RandomMemory(
        memory_size=25000,
        num_envs=n_envs,
        device=device,
        replacement=False,
    )
    models = {
        "policy": TD3Policy(env.observation_space, env.action_space, device),
        "target_policy": TD3Policy(env.observation_space, env.action_space, device),
        "critic_1": TD3Value(env.observation_space, env.action_space, device),
        "critic_2": TD3Value(env.observation_space, env.action_space, device),
        "target_critic_1": TD3Value(env.observation_space, env.action_space, device),
        "target_critic_2": TD3Value(env.observation_space, env.action_space, device),
    }
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = copy.deepcopy(TD3_DEFAULT_CONFIG)
    cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
    cfg["smooth_regularization_clip"] = 0.5
    cfg["batch_size"] = 100
    cfg["random_timesteps"] = 100
    cfg["learning_starts"] = 100
    cfg["experiment"]["write_interval"] = ALGO_CONFIGS["td3"].write_interval
    cfg["experiment"]["checkpoint_interval"] = ALGO_CONFIGS[
        "td3"
    ].checkpoint_interval
    cfg["experiment"]["directory"] = experiment_dir

    return TD3(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )


AGENT_BUILDERS = {
    "ppo": _build_ppo,
    "td3": _build_td3,
}


def train(
    *,
    algo: str = "ppo",
    env_name: str = "reach",
    num_steps: int | None = None,
    n_envs: int | None = None,
    render_mode: str | None = None,
    robot_preset: str | None = None,
    seed: int | None = None,
    experiment_dir: str | None = None,
    log_interval: int = 100,
    metrics_window: int = 100,
    console_metrics: bool = True,
) -> None:
    """Train an SKRL agent on a configured DexSuite environment."""
    validate_env_algo(env_name, algo)
    algo_cfg = ALGO_CONFIGS[algo]
    env_cfg = ENV_CONFIGS[env_name]
    train_steps = int(num_steps if num_steps is not None else algo_cfg.default_timesteps)
    train_envs = int(n_envs if n_envs is not None else env_cfg.default_n_envs)

    if seed is None:
        set_seed()
    else:
        set_seed(seed)

    env = make_skrl_env(
        env_name,
        n_envs=train_envs,
        render_mode=render_mode,
        robot_preset=robot_preset or env_cfg.default_robot_preset,
    )
    env = EpisodeStatsMonitor(
        env,
        log_interval=log_interval,
        window_size=metrics_window,
        print_stats=console_metrics,
    )
    env.reset()
    device = env.device

    agent = AGENT_BUILDERS[algo](
        env,
        device=device,
        n_envs=train_envs,
        experiment_dir=experiment_dir or algo_cfg.experiment_dir,
    )
    trainer = SequentialTrainer(
        cfg={"timesteps": train_steps, "headless": render_mode != "human"},
        env=env,
        agents=[agent],
    )
    trainer.train()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Train SKRL agents on DexSuite.")
    parser.add_argument("--algo", choices=algo_choices(), default="ppo")
    parser.add_argument("--env", dest="env_name", choices=env_choices(), default="reach")
    parser.add_argument("--num-steps", "--num_steps", type=int, default=None)
    parser.add_argument("--n-envs", "--n_envs", type=int, default=None)
    parser.add_argument(
        "--render-mode",
        "--render_mode",
        choices=["human", "rgb_array"],
        default=None,
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print and track train metrics every N environment steps.",
    )
    parser.add_argument(
        "--metrics-window",
        type=int,
        default=100,
        help="Number of completed episodes in the rolling reward/success window.",
    )
    parser.add_argument(
        "--no-console-metrics",
        action="store_true",
        help="Disable periodic console reward/success prints.",
    )
    return parser


def main() -> None:
    """Run training from CLI arguments."""
    args = build_parser().parse_args()
    train(
        algo=args.algo,
        env_name=args.env_name,
        num_steps=args.num_steps,
        n_envs=args.n_envs,
        render_mode=args.render_mode,
        robot_preset=args.robot_preset,
        seed=args.seed,
        experiment_dir=args.experiment_dir,
        log_interval=args.log_interval,
        metrics_window=args.metrics_window,
        console_metrics=not args.no_console_metrics,
    )


if __name__ == "__main__":
    main()
