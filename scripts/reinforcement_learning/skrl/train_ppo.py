"""Compatibility entrypoint for PPO training."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.reinforcement_learning.skrl.train import build_parser, train
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.reinforcement_learning"}:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import build_parser, train


def main() -> None:
    """Run PPO training with the shared SKRL trainer."""
    parser = build_parser()
    parser.set_defaults(algo="ppo")
    args = parser.parse_args()
    train(
        algo="ppo",
        env_name=args.env_name,
        num_steps=args.num_steps,
        n_envs=args.n_envs,
        render_mode=args.render_mode,
        robot_preset=args.robot_preset,
        seed=args.seed,
        experiment_dir=args.experiment_dir,
    )


if __name__ == "__main__":
    main()
