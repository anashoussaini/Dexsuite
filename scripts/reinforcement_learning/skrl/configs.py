"""Central configuration registry for DexSuite SKRL experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Static metadata for a trainable DexSuite environment."""

    name: str
    task: str
    wrapper: str
    supported_algorithms: tuple[str, ...]
    default_n_envs: int
    default_robot_preset: str
    control_hz: int = 20
    cameras: None = None


@dataclass(frozen=True)
class AlgoConfig:
    """Default training knobs for an SKRL algorithm."""

    name: str
    default_timesteps: int
    default_n_envs: int
    experiment_dir: str = "runs"
    write_interval: int = 200
    checkpoint_interval: int = 1000


ENV_CONFIGS: dict[str, EnvConfig] = {
    "reach": EnvConfig(
        name="reach",
        task="reach",
        wrapper="reach",
        supported_algorithms=("ppo", "td3"),
        default_n_envs=64,
        default_robot_preset="franka_with_gripper_normalized",
    ),
    "lift": EnvConfig(
        name="lift",
        task="lift",
        wrapper="lift",
        supported_algorithms=("ppo",),
        default_n_envs=256,
        default_robot_preset="franka_umi_tactile_joint_position",
    ),
}

ALGO_CONFIGS: dict[str, AlgoConfig] = {
    "ppo": AlgoConfig(
        name="ppo",
        default_timesteps=100_000,
        default_n_envs=64,
        write_interval=200,
        checkpoint_interval=1000,
    ),
    "td3": AlgoConfig(
        name="td3",
        default_timesteps=50_000,
        default_n_envs=16,
        write_interval=100,
        checkpoint_interval=5000,
    ),
}


def env_choices() -> tuple[str, ...]:
    """Return supported environment names for CLI choices."""
    return tuple(sorted(ENV_CONFIGS))


def algo_choices() -> tuple[str, ...]:
    """Return supported algorithm names for CLI choices."""
    return tuple(sorted(ALGO_CONFIGS))


def validate_env_algo(env_name: str, algo: str) -> None:
    """Raise when an algorithm is not configured for an environment."""
    env_cfg = ENV_CONFIGS[env_name]
    if algo not in env_cfg.supported_algorithms:
        supported = ", ".join(env_cfg.supported_algorithms)
        raise ValueError(
            f"Algorithm '{algo}' is not configured for '{env_name}'. "
            f"Supported algorithms: {supported}.",
        )
