<p align="center">
  <img src="https://dexsuite.readthedocs.io/en/latest/_static/brand/wordmark.svg" alt="DexSuite" width="360">
</p>

<h3 align="center">Dexterous manipulation. Simulated, teleoperated, benchmarked.<br>All in one framework.</h3>

<p align="center">
  <a href="https://dexsuiteorg.github.io/"><b>Website</b></a> ·
  <a href="https://dexsuite.readthedocs.io/"><b>Documentation</b></a> ·
  <a href="https://huggingface.co/dexsuite"><b>Datasets</b></a> ·
  <a href="https://arxiv.org/abs/2610.37799"><b>Paper</b></a> ·
  <a href="https://github.com/anashoussaini/Dexsuite"><b>Code</b></a>
</p>

<p align="center"><sub>DexSuite is in beta: the API may still change before 1.0. Pin the version for reproducible work, and please
<a href="https://github.com/anashoussaini/Dexsuite/issues">open an issue</a> if anything breaks.</sub></p>

<p align="center">
  <img src="https://dexsuite.readthedocs.io/en/latest/_static/home/hero.webp" alt="A Sharpa hand picks up a spray can and paints the stencil" width="480">
</p>

DexSuite brings the robot, the tasks, teleoperation, tactile sensing, the data and the evaluation into one stack, on the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator. A task is written once and runs with any arm and
any hand, from a parallel-jaw gripper to a 24-joint hand.

## Install

```bash
pip install dexsuite                # every arm and hand, every task but Close Window and Spray (clone the repo for it)
pip install "dexsuite[all]"         # plus SpaceMouse, Vive and Manus teleoperation and the LeRobot dataset tools
```

Python 3.10 to 3.13 and a CUDA GPU (the CPU works for a quick look). On Linux, pip builds `evdev` for the keyboard
backend: install a C compiler and the Python headers first (`sudo apt install build-essential python3-dev`). The first environment on a machine compiles the
Genesis kernels, which takes about a minute. On a machine without a display, `export PYOPENGL_PLATFORM=egl`.

| | |
|---|---|
| Tasks | 31: Reach, Lift, Push, Pick & Place, Stack, Rack Mug, Move Pan, Drill to Point, Place Fruits, Spray Graffiti, Toast Bread, Make Coffee, Tool Hang, Duck in Bowl, Open Safe, Stow in Safe, Pliers in Jewelry Box, Move Pan to Platform, Close Window and Spray and the twelve bimanual tasks |
| Arms | Franka Panda, Franka FR3, Kinova Gen3, KUKA iiwa 14, Rethink Sawyer, UR5e, UR10e, UFactory Lite 6, Link6, Unitree Z1, six integrated robots and a floating wrist for hand-only tasks |
| End-effectors | Robotiq 2F-85, UMI gripper, Allegro Hand, BarrettHand, D'Claw, Inspire, Ability, Schunk SVH, LEAP and Shadow hands, and the Sharpa Wave hand on the floating wrist |
| `[all]` | SpaceMouse, Vive and Manus teleoperation, LeRobot recording (0.4.4), HDF5 export |

## Quickstart

```python
import dexsuite as ds

env = ds.make("lift", manipulator="franka", gripper="allegro")
obs, info = env.reset(seed=0)

for _ in range(300):
    action = env.action_space.sample()       # 6 values move the hand, 16 set the Allegro joints
    obs, reward, terminated, truncated, info = env.step(action)
```

Every environment is a Gymnasium environment with batched PyTorch tensors: `n_envs=64` builds 64 copies in one
simulator. The [documentation](https://dexsuite.readthedocs.io/) covers the tasks, the robots, the actions and
observations, teleoperation and the benchmark datasets on [Hugging Face](https://huggingface.co/dexsuite).

## Citation

```bibtex
@article{houssaini2026dexsuite,
  title   = {DexSuite: An All-in-One Framework for Dexterous Manipulation Across Arms, Hands, and Tasks},
  author  = {Houssaini, Anas and He, Hugo and Shi, Junming and Abyaneh, Amin and Chahine, Ricardo and Wen, Shuo and Danesh, Mohamad H. and Fr{\'e}chette, Charles-{\'E}tienne and Sosa Guzman, Mariana and Desrochers, Mathias and Houssaini, Rayan and Landa, Adam and Soulie, Theophile and Morissette, Charlotte and Tims, Annie and Russi, Martino and Lussier, Jonathan and Dudek, Gregory and Precup, Doina and Urain, Julen and Lin, Hsiu-Chin and Meger, David},
  journal = {arXiv preprint arXiv:2610.37799},
  year    = {2026}
}
```

## License

The DexSuite code is released under the MIT License. The robot, hand and object models keep their own
licences, listed in THIRD_PARTY_LICENSES.md. Some objects (the PartNet-Mobility models) are for non-commercial
research and education only.
