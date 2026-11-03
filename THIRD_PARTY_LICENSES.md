# Third-party models and code

The DexSuite code is released under the MIT License (`LICENSE`). The robot, hand and object models in
`dexsuite/models` come from other projects and keep their own licences, listed here. Where a model folder carries a
`LICENSE`, `NOTICE` or `ATTRIBUTION.txt` file, that file is the reference. Third-party code is listed under
[Code](#code), at the end.

Some object models are licensed for **non-commercial research and education only** (PartNet-Mobility, below). Check
the licence of every model you use before any commercial use.

## Arms (`dexsuite/models/manipulators`)

| Folder | Model | Source | Licence |
|---|---|---|---|
| `franka` | Franka Emika Panda | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Apache-2.0 |
| `franka_fr3` | Franka Research 3 | MuJoCo Menagerie | Apache-2.0 |
| `gen3`, `gen3_lite`, `link6` | Kinova Gen3, Gen3 lite, Link 6 | MuJoCo Menagerie | BSD-3-Clause (Kinova) |
| `kuka` | KUKA LBR iiwa 14 | MuJoCo Menagerie | BSD-3-Clause (Robot Locomotion Group, MIT CSAIL) |
| `lite6` | UFACTORY Lite 6 | MuJoCo Menagerie | BSD-3-Clause (UFACTORY) |
| `sawyer` | Rethink Robotics Sawyer | MuJoCo Menagerie | Apache-2.0 |
| `so_100` | SO-ARM100 | MuJoCo Menagerie | Apache-2.0 |
| `trossen_vx250s`, `trossen_vx300s`, `aloha_tactile` | Trossen WidowX 250 s, ViperX 300 s (ALOHA) | MuJoCo Menagerie | BSD-3-Clause (Trossen Robotics) |
| `ur5`, `ur10` | Universal Robots UR5e, UR10e | MuJoCo Menagerie | BSD-3-Clause (ROS-Industrial) |
| `z1` | Unitree Z1 | MuJoCo Menagerie | BSD-3-Clause (Unitree Robotics) |
| `floating` | floating wrist | DexSuite | MIT |

## Hands and grippers (`dexsuite/models/grippers`)

| Folder | Model | Source | Licence |
|---|---|---|---|
| `ability` | PSYONIC Ability Hand | [dex-urdf](https://github.com/dexsuite/dex-urdf), from PSYONIC's URDF | upstream notice in `assets/LICENSE.txt` |
| `allegro` | Wonik Allegro Hand | MuJoCo Menagerie | BSD-2-Clause (SimLab) |
| `barrett` | Barrett Hand | [dex-urdf](https://github.com/dexsuite/dex-urdf) | BSD-2-Clause (Barrett Technology) |
| `dclaw` | ROBEL D'Claw | [dex-urdf](https://github.com/dexsuite/dex-urdf) | Apache-2.0 (The ROBEL Authors) |
| `inspire` | Inspire Hand | [dex-urdf](https://github.com/dexsuite/dex-urdf), from Inspire's CAD | upstream notice in `assets/LICENSE.txt` |
| `leap` | LEAP Hand | MuJoCo Menagerie | MIT (Ananye Agarwal) |
| `robotiq` | Robotiq 2F-85 | MuJoCo Menagerie | BSD-2-Clause (ROS-Industrial) |
| `schunk` | Schunk SVH | [dex-urdf](https://github.com/dexsuite/dex-urdf) | GPL-3.0 |
| `shadow` | Shadow Dexterous Hand | MuJoCo Menagerie | Apache-2.0 |
| `sharpa` | Sharpa Wave | [Sharpa SDK](https://github.com/sharpa-robotics/sharpa-vision-retargeting-sdk) | Apache-2.0 (Sharpa Group), modified files listed in `assets/NOTICE` |
| `umi` | UMI gripper with tactile pads | DexSuite authors (fingers and pads), on the Franka hand body of MuJoCo Menagerie's Panda | MIT (fingers and pads), Apache-2.0 (hand body, `manipulators/franka`) |

## Scene (`dexsuite/models/arena`)

| Folder | Model | Source | Licence |
|---|---|---|---|
| `assets/vention_table` | Vention table | [furniture_sim](https://github.com/vikashplus/furniture_sim) by Vikash Kumar | Apache-2.0 |

## Objects (`dexsuite/models/objects`)

**YCB objects.** The scanned objects with `textured.obj` and `texture_map.png` (fruit, boxes and cans in `food`, the
kitchen utensils, the tools, the balls, cups, blocks and toys in `shape` and `task`) come from the
[YCB Object and Model Set](https://www.ycbbenchmarks.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). DexSuite adds collision decompositions and MJCF/URDF
wrappers. Please cite B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel and A. M. Dollar, "Benchmarking in
Manipulation Research: The YCB Object and Model Set and Benchmarking Protocols", IEEE Robotics and Automation
Magazine, 2015.

**PartNet-Mobility objects** (`partnet/`). From the [SAPIEN PartNet-Mobility dataset](https://sapien.ucsd.edu/),
under its terms of use: **non-commercial research and education only**. Please cite F. Xiang et al., "SAPIEN: A
SimulAted Part-based Interactive ENvironment", CVPR 2020.

**Artiverse objects.** `kitchen/toaster_smeg` and `tools/aerosol_can` are articulated models from
[Artiverse](https://huggingface.co/datasets/3dlg-hcvc/artiverse), built on Objaverse models under CC BY 4.0. Authors
and links are in each folder's `ATTRIBUTION.txt`.

**More Artiverse objects.** `task/safe`, `task/window`, `task/jewelry_box` and `tools/pliers` are articulated models from
[Artiverse](https://huggingface.co/datasets/3dlg-hcvc/artiverse) (the window, the jewelry box and the pliers from its 3D Warehouse
source, the safe from its Objaverse-XL source), used under the Artiverse dataset terms for research.
`tools/spray_bottle_blue` comes from Artiverse's Objaverse-XL source, originally a GPL-3.0 repository (licence text in the folder).
Sources are in each folder's `ATTRIBUTION.txt`.

**Genesis.** `task/duck` is the rubber duck bundled with [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
(Apache-2.0).

**Sketchfab models.** `food/toast` (CC BY-SA 4.0) and `kitchen/butter_knife` (CC BY 4.0). Authors and links are in
each folder's `ATTRIBUTION.txt`.

**Vikash Kumar's object_sim.** `kitchen/bowl` from [object_sim](https://github.com/vikashplus/object_sim),
Apache-2.0.

## Code

| Files | Source | Licence |
|---|---|---|
| `scripts/imitation_learning/robomimic/custom_algorithms/act.py`, `act_nets.py`, `act_config.py` | [ACT](https://github.com/tonyzhaozh/act) by Tony Z. Zhao, whose model code is modified from [DETR](https://github.com/facebookresearch/detr) (Facebook, Inc. and its affiliates), re-implemented on robomimic by the DexSuite authors | MIT (ACT) and Apache-2.0 (DETR), texts in the folder's `LICENSE-ACT` and `LICENSE-DETR`, changes listed in its `NOTICE` |
