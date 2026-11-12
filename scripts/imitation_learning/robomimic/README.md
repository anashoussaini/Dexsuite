# DexSuite robomimic Training

Use `train.py` as the single entrypoint for dataset download, LeRobot to HDF5
conversion, robomimic config generation, training, and optional rollouts. Run
every command from the repository root.

## Install

The training scripts live in the repository, so install from a clone:

```bash
git clone https://github.com/anashoussaini/Dexsuite && cd Dexsuite
pip install -e ".[il]"
python -m scripts.imitation_learning.robomimic.install
```

The second line brings LeRobot and robomimic's runtime dependencies. The third
installs robomimic 0.5.0 from its GitHub tag without robomimic's own version
pins, which would break LeRobot (PyPI only has robomimic 0.3.0, which these
recipes do not run on). It works in pip and uv environments. The first import of
robomimic 0.5.0 downloads a 1.6 GB CLIP model into `$HF_HOME`.

Then train, for example BC on the human demonstrations of Lift with the Allegro
hand:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config bc \
  --sensors state
```

## Datasets

`--dataset` selects what to train on:

- A name from `config/datasets.yaml`: every set on
  [huggingface.co/dexsuite](https://huggingface.co/dexsuite) is registered. That's
  the 16 benchmark pairs (`reach_franka_robotiq`, `lift_franka_allegro`,
  `spray_graffiti_floating_sharpa_scripted`, ...), their machine-generated
  `_mg` sets, and the scripted controller sets. `--list-datasets` prints them.
- Any Hugging Face dataset repo id in the DexSuite LeRobot format, e.g.
  `--dataset someone/my_lift_demos`. The run is named after the repo, and the
  task is read from the dataset's `meta/env_config.json` (or pass `--task`).
- A local LeRobot folder recorded with DexSuite.
- A converted `.hdf5` file.

`--config` selects the policy recipe: `bc` for behavior cloning (`gmm_best`,
`gaussian_best`, `bc_rnn` and `bc_transformer` are the tuned BC variants),
`act` or `act_img` for Action Chunking Transformer, or `diffusion` for diffusion
policy. The recipe is not tied to cameras or low dimensional inputs.

## Policy Shape

`--sensors` selects the observation source:

- `state` uses the task low-dimensional observation profile.
- `proprio` uses robot-obtainable low-dimensional keys only.
- `cameras` uses selected camera images plus robot proprio by default.

Without `--sensors`, the dataset entry's `default_sensors` decides. That is
`state` for the registered sets and for any repo id or folder outside the
registry.

`--architecture` selects the BC model family:

- `mlp` for simple state or proprio policies.
- `rnn` for temporal low-dimensional policies.
- `transformer` for longer-context BC policies.

Camera policies can also set:

- `--cameras front wrist`, only keep matching camera observations.
- `--backbone ResNet18Conv`, choose the robomimic visual backbone.
- `--pretrained` or `--no-pretrained`, control backbone initialization.

## ACT

ACT is DexSuite's robomimic port of Action Chunking with Transformers (Zhao et
al., RSS 2023) in `custom_algorithms/`, the code behind the paper's ACT
results. It runs on stock robomimic 0.5. From the newest observation it
predicts a chunk of future actions, trains on L1 + 10 x KL with a CVAE latent
(z = 0 at test time) and blends overlapping chunks in rollouts with ACT's
temporal ensembling (k = 0.01). Images go through ACT's own ResNet18 (ImageNet
weights, frozen batch norm, 0.1 x the learning rate), so `--backbone`,
`--pretrained` and `--crop-size` do not apply. `act` is the low dimensional
recipe. `act_img` is the paper's image protocol, with a 196 px crop (random in
training, centred at test).

ACT policies can also set:

- `--action-horizon 8` (alias `--chunk-size`), the chunk length.
- `--obs-horizon 1`, frames loaded per sample. ACT only reads the newest one.
- `--act-embed-dim 512`, transformer width.
- `--act-num-layers 6`, decoder layers (the observation encoder has 4).
- `--act-num-heads 8`, attention heads.
- `--act-dropout 0.1`, encoder and decoder dropout.
- `--act-lr 1e-4`, learning rate (the image backbone trains at 0.1 x).

`train.frame_stack` is kept equal to `obs_horizon` and `train.seq_length` to
`action_horizon`.

Unlike the original ACT code, this port uses every decoder layer (the original
keeps the first layer's output), fixed sinusoidal action queries (learned
there), GELU in the observation encoder and decoder (ReLU there), no weight
decay, and robomimic's chunk padding, which repeats the last action where the
original masks padded steps out of the loss.

## Rollouts

`--rollout standard` is used by default with managed recipes. Use
`--rollout off` for pure training or prepare-only checks, and use
`--rollout smoke` for cheaper validation.

Rollout horizon comes from the rollout preset and environment config, not from
the training command. `--debug` shortens training and rollout settings only for
local smoke tests.

## Useful Overrides

- `--name EXPERIMENT`, set the robomimic experiment name.
- `--prepare-only`, download, convert, and write config without training.
- `--force-download`, refresh the LeRobot dataset.
- `--force-convert`, rebuild the HDF5 file.
- `--read-workers N`, parallel readers for the conversion (default 1 when camera
  videos are decoded, since the video decoder is not thread safe, 8 otherwise).
- `--action-normalization dataset|none|min_max|gaussian`, override metadata.
- `--output-dir PATH`, write robomimic runs somewhere else.
- `--list-datasets`, show configured dataset names.
- `--list-policy-presets`, show policy recipes.
- `--list-rollout-presets`, show rollout presets.

## Generated Files

Managed runs create this layout:

```text
datasets/<task>/lerobot/<repo-or-run>/
datasets/<task>/hdf5/<dataset>.hdf5
datasets/<task>/hdf5/action_config.json
datasets/<task>/hdf5/obs_*.yaml
datasets/<task>/configs/<experiment>.json
runs/imitation_learning/robomimic/<task>/
```

`action_config.json` carries the action layout and normalization metadata from
the dataset conversion. Generated `obs_*.yaml` files expose robomimic's required
`obs.low_dim`, `obs.rgb`, and `obs.depth` keys, plus `groups.proprio` and
`groups.state` so robot observations and environment state remain explicit.

## Commands

List the integrated datasets:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --list-datasets
```

Prepare the low-dimensional `lift_franka_allegro` HDF5 and robomimic config:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config bc \
  --sensors state \
  --architecture mlp \
  --rollout off \
  --prepare-only
```

Train low-dimensional BC with the reference settings, rollouts enabled, and no
model checkpoint writes:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config bc \
  --sensors state \
  --architecture mlp \
  --rollout standard \
  --name lift_franka_allegro_bc_lowdim \
  --output-dir ../bc_trained_models \
  --num-epochs 200 \
  --batch-size 100 \
  --epoch-every-n-steps 100 \
  --validation-epoch-every-n-steps 10 \
  --hdf5-normalize-obs \
  --train-dataset-keys actions rewards dones \
  --lr 0.0001 \
  --no-bc-gaussian \
  --no-save-models \
  --render-video \
  --video-skip 5 \
  --rollout-enabled \
  --rollout-n 5 \
  --rollout-horizon 400 \
  --rollout-rate 25 \
  --rollout-warmstart 0 \
  --rollout-terminate-on-success
```

Prepare camera data and config. This rebuilds the HDF5 if the existing file is
low-dimensional only:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config bc \
  --sensors cameras \
  --cameras front wrist \
  --architecture mlp \
  --rollout off \
  --force-convert \
  --prepare-only
```

Train camera BC with front and wrist cameras:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config bc \
  --sensors cameras \
  --cameras front wrist \
  --architecture transformer \
  --backbone ResNet18Conv \
  --pretrained \
  --rollout smoke
```

Train ACT on robot proprio:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset lift_franka_allegro \
  --config act \
  --sensors proprio \
  --action-horizon 8 \
  --rollout off
```

Train image ACT with the paper's protocol: front camera and robot proprio
(`config/act_img_front_obs.yaml`), chunks of 40, 300 epochs of 100 steps, 50
rollouts and a checkpoint every 50 epochs:

```bash
python -m scripts.imitation_learning.robomimic.train \
  --dataset reach_franka_robotiq \
  --h5-basename reach_franka_robotiq_img \
  --config act_img \
  --sensors cameras \
  --cameras front \
  --obs scripts/imitation_learning/robomimic/config/act_img_front_obs.yaml \
  --obs-horizon 1 \
  --action-horizon 40 \
  --num-epochs 300 \
  --rollout standard \
  --rollout-rate 50 \
  --no-render-video \
  --save-models \
  --seed 0
```
