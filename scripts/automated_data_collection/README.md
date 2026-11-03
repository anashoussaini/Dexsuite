# Automated Data Collection

This folder contains scripted collectors for DexSuite single-arm tasks. The
direct collectors use `osc_pose_abs_quat` for the arm and `joint_position` for
the gripper. The sent action is the flat action vector passed to
`env.step(action)`, not a read-back observation.

## Scripts

| Script | Task | Notes |
| --- | --- | --- |
| `reach_automated.py` | `reach` | Uses the reusable primitive planner and runner. |
| `lift_automated.py` | `lift` | Direct TCP-to-wrist target script with optional tracing. |
| `pick_place_automated.py` | `pick_place` | Direct grasp, transport, release, retreat script. |
| `push_automated.py` | `push` | Direct face-selection push script. |
| `stack_automated.py` | `stack` | Direct two-cube stack script. |
| `benchmark_baselines.py` | all five | Runs success and smoothness checks and writes CSV summaries. |

## Action Convention

For the default `franka + umi` setup the action has 9 values:

```text
[0:3] wrist position delta used by osc_pose_abs_quat
[3:7] wrist quaternion delta in WXYZ order
[7:9] gripper joint-position command
```

The scripts plan in TCP space, then convert TCP targets to wrist targets before
sending the final action to the environment. This is why the trace files include
both target columns and `arm_action_*` / `grip_action_*` columns.

## Running

Run from the repository root with the simulator environment:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.lift_automated \
  --episodes 10 --no-record --render-mode none --cameras none
```

Use `--viewer` for a live run:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.lift_automated \
  --episodes 10 --no-record --viewer --cameras none
```

## Smooth Grasp-to-Motion Transition

Lift uses these shorter defaults to avoid overrepresenting repeated reach and
lift actions in BC data:

```text
--contact-min-alpha 1.0
--move-to-cube-steps 100
--close-steps 5
--settle-steps 2
--lift-steps 50
--hold-steps 10
--lift-start-progress 0.0
--transport-start-progress 0.0
```

At the default 20 Hz control rate, lift now records at most 167 actions per
successful episode before early termination: 100 reach, 5 close, 2 settle, 50
lift, and 10 hold. Transport and lift start at zero trajectory progress, so the
first motion target is continuous with the grasp target.

## Keyboard Gripper Override

The direct collectors and shared runner can keep the scripted arm trajectory
while replacing only the gripper action slice from the keyboard:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.lift_automated \
  --episodes 1 --no-record --viewer --cameras none \
  --gripper-mode keyboard
```

Use `o` / `p` to adjust the gripper while the trajectory continues. The
keyboard state is asynchronous and non-blocking; each control step polls the
latest gripper command and maps its alpha into the normal gripper action
format. For the default UMI gripper, alpha `0` sends `[1, -1]` and alpha `1`
sends `[-1, 1]`.

Useful knobs:

```text
--keyboard-layout qwerty
--keyboard-gripper-speed 1.0
--keyboard-gripper-initial-alpha 0.0
--keyboard-gripper-invert
```

`--keyboard-gripper-invert` flips the sign if the open/close direction feels
backwards on the local keyboard mapping.

## Target Noise

All collectors expose shared low-frequency target-noise knobs. Defaults are off:

```text
--target-noise-xy-std 0.0
--target-noise-z-std 0.0
--target-noise-hold-steps 12
--target-noise-max-xy 0.015
--target-noise-max-z 0.010
--target-noise-phases lift,transport,retreat
--target-noise-seed <optional int>
```

Example:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.lift_automated \
  --episodes 10 --no-record --render-mode none --cameras none \
  --target-noise-xy-std 0.003 \
  --target-noise-z-std 0.002 \
  --target-noise-hold-steps 10
```

Noise is applied to commanded Cartesian targets before action conversion. Each
sampled offset is held for several control steps, which creates smooth action
diversity instead of per-step jitter. Final samples of line and arc segments are
kept exact so release, retreat, and placement endpoints stay well-defined.

Useful phase filters:

```text
--target-noise-phases lift
--target-noise-phases transport,retreat
--target-noise-phases push
--target-noise-phases reach
--target-noise-phases all
```

## Tracing Lift Actions

`lift_automated.py` can write detailed target/action traces:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.lift_automated \
  --episodes 1 --max-attempts 1 --no-record \
  --render-mode none --cameras none \
  --trace-dir /tmp/dexsuite_lift_action_trace \
  --trace-prefix lift_trace
```

The trace directory contains:

```text
trajectory.csv
metadata.json
trajectory_summary.png
```

## Validation

Run Ruff on the full folder:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m ruff check scripts/automated_data_collection
```

Compile the entrypoints:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m py_compile \
  scripts/automated_data_collection/reach_automated.py \
  scripts/automated_data_collection/lift_automated.py \
  scripts/automated_data_collection/pick_place_automated.py \
  scripts/automated_data_collection/push_automated.py \
  scripts/automated_data_collection/stack_automated.py \
  scripts/automated_data_collection/benchmark_baselines.py \
  scripts/automated_data_collection/planner.py \
  scripts/automated_data_collection/utils/runner.py \
  scripts/automated_data_collection/utils/cli.py \
  scripts/automated_data_collection/utils/keyboard_gripper.py \
  scripts/automated_data_collection/utils/target_noise.py
```

Run 10 episodes for every automated task and write success/smoothness summaries:

```bash
/usr/local/data/anas/miniconda3/envs/robomimic_venv/bin/python \
  -m scripts.automated_data_collection.benchmark_baselines \
  --episodes 10 --tasks reach,lift,pick_place,push,stack --print-every 0
```
