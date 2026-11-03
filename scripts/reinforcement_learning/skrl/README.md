# DexSuite SKRL Training

This directory contains the SKRL integration for DexSuite reinforcement learning.
Use `train.py` as the main entrypoint and pass the algorithm with `--algo`.

## Quick Start

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  scripts/reinforcement_learning/skrl/train.py \
  --algo ppo \
  --env reach \
  --n-envs 64 \
  --num-steps 100000
```

Short smoke test:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  scripts/reinforcement_learning/skrl/train.py \
  --algo ppo \
  --env reach \
  --n-envs 2 \
  --num-steps 2
```

Lift PPO training:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  scripts/reinforcement_learning/skrl/train.py \
  --algo ppo \
  --env lift \
  --robot-preset franka_umi_tactile_joint_position \
  --n-envs 256 \
  --num-steps 1000000 \
  --log-interval 100 \
  --seed 0
```

The trainer prints rolling metrics while it runs:
`reward_step`, completed-episode reward, `success_rate`, episode length,
completed episodes, and env steps per second. These values are also attached to
SKRL's `episode/*` info stream for experiment logging.

Compatibility entrypoints still work:

```bash
python scripts/reinforcement_learning/skrl/train_ppo.py --n-envs 64
python scripts/reinforcement_learning/skrl/train_td3.py --n-envs 16
```

## Supported Matrix

| Env | Algorithms | Status | Notes |
| --- | --- | --- | --- |
| `reach` | `ppo`, `td3` | Tested | Low-dimensional state wrapper, parallel envs, `cameras=None`, reset-by-index. PPO is the default recommendation. TD3 works as an off-policy baseline. |
| `lift` | `ppo` | Tested smoke | Franka + UMI tactile preset, compact tactile contact/force observations, staged lift reward. PPO is preferred here because the reward is dense and phase-like. |

## Layout

| File | Purpose |
| --- | --- |
| `configs.py` | Registry of supported envs and algorithms. Add new trainable combos here first. |
| `envs.py` | DexSuite env construction and SKRL wrapping. Keeps `cameras=None` for current low-dimensional RL. |
| `models.py` | PPO and TD3 model classes. |
| `utils.py` | Shared checkpoint and action-mapping helpers. |
| `wrappers.py` | SKRL environment wrappers. |
| `train.py` | Unified trainer. Pass `--algo ppo` or `--algo td3`. |
| `eval_ppo.py` | PPO checkpoint evaluation using the same centralized env/model helpers. |
| `benchmark_capacity.py` | Power-of-two vectorization sweep with reset/step timing, memory, CUDA, reset-by-index, and randomization checks. |

## Capacity Benchmark

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  scripts/reinforcement_learning/skrl/benchmark_capacity.py \
  --env lift \
  --robot-preset franka_umi_tactile_joint_position \
  --max-envs 2048 \
  --steps 2 \
  --output /tmp/dexsuite_lift_capacity.csv
```

## Robot Presets

Default:

```bash
--robot-preset franka_with_gripper_normalized
```

This matches the PPO setup and gives the reach env a 9D action space.

Legacy:

```bash
--robot-preset legacy_franka_robotiq
```

This matches the old TD3 script and gives the reach env an 8D action space.

Lift tactile:

```bash
--env lift --robot-preset franka_umi_tactile_joint_position
```

This uses Franka with the `umi` gripper, normalized
`joint_position` arm and gripper control, compact tactile contact summaries,
and `cameras=None`.

For lift, the policy still outputs normalized actions, but the wrapper maps the
arm slice like Isaac Lab's Franka lift joint-position setup: actions are treated
as scaled offsets around the Franka home pose, then converted back to normalized
joint-position targets. DexSuite's reset pose starts farther from the cube than
Isaac Lab's Franka lift layout, so the arm offset scale is `2.0 rad` with a
`0.15 rad` per-step slew limit. The UMI gripper slice is mapped like a binary
open/close command: open is `[1, -1]`, close is `[-1, 1]`.

## Evaluation

```bash
python scripts/reinforcement_learning/skrl/eval_ppo.py \
  --checkpoint runs/<run>/checkpoints/best_agent.pt \
  --env reach \
  --n_envs 4 \
  --episodes 256
```

If a checkpoint was trained with the legacy 8D action space and the current env
uses 9D actions, `eval_ppo.py` can adapt one trailing action dimension with:

```bash
--action_map auto
```

## Lift Reward

The `lift` reward follows Isaac Lab's Franka cube-lift structure, adapted for
DexSuite and UMI tactile:

- Reach object: weak broad shaping `0.25 * (1 - tanh(tcp_to_grasp / 0.3))`
  plus stronger fine shaping `2.0 * (1 - tanh(tcp_to_grasp / 0.08))`.
- Lift object: `+15` once the cube is at least `4 cm` above the table.
- Goal tracking after lift: coarse `1 - tanh(cube_to_goal / 0.3)` with weight
  `16`, plus fine `1 - tanh(cube_to_goal / 0.05)` with weight `5`.
- Near-object close-command bridge reward gated by fine reach, for learning
  contact with the structured UMI gripper action.
- Penalties for action rate, joint velocity, and time.

The tactile observation uses compact summaries from the UMI pads:
`left/right contact` and `left/right force_norm`. The policy observes those
features through `DexsuiteLiftWrapper`, along with relative
`grasp_position - tcp_position` and `target_position - cube_position` vectors.
The reward does not require cameras.
