# Contributing to DexSuite

Thanks for helping. New tasks, robots, devices and fixes are all welcome.

## Set up

```bash
git clone https://github.com/anashoussaini/Dexsuite.git
cd Dexsuite
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,il]" pytest ruff roboticstoolbox-python "skrl>=1.4,<2"
python -m scripts.imitation_learning.robomimic.install   # robomimic 0.5.0, for the imitation learning tests
```

On a machine without a display, `export PYOPENGL_PLATFORM=egl` before running anything that renders.

## Before you open a pull request

```bash
pytest tests/config -q                  # fast: configs, the task registry, YAML files
pytest tests -q                         # everything, builds robots and environments on the GPU
ruff check path/to/changed_file.py      # lint the files you touched
ruff format path/to/changed_file.py
```

The lint and format rules live in `pyproject.toml`. Older files do not all pass them yet, so check the files you change
rather than the whole tree.

## Add a task

A task subclasses `BaseEnv` in `dexsuite/environments/`, fills in its hooks (the scene, the reset, the extra
observations, the reward, the success and failure checks) and registers itself:

```python
from dexsuite.core.registry import register_env

@register_env("my_task")
class MyTask(BaseEnv):
    ...
```

[Add a task](https://dexsuite.readthedocs.io/en/latest/how-to/new-task.html) walks through a complete one. A task is
ready when it builds with a gripper and with a hand, its success check fires on a scripted or teleoperated rollout,
and, if it batches, a partial reset with `n_envs > 1` works.

## Add an arm or an end-effector

Each arm lives in `dexsuite/models/manipulators/<arm>/` with its model files, a Python class registered with
`register_manipulator`, a home pose and an adapter file, `<arm>_adapters.yaml`, that mounts every end-effector it
supports on its flange. End-effectors live in `dexsuite/models/grippers/<name>/`.

For an arm whose flange differs from an existing one only by a fixed transform, derive its adapters instead of writing
them by hand: add it to `ARMS` in `scripts/assets/make_adapters.py` and run

```bash
python scripts/assets/make_adapters.py
```

Add its workspace box to `dexsuite/config/workspaces.yaml`, then check that the end-effector points down at the table
from the home pose and that the wrist camera sees the workspace. The how-to pages cover
[arms](https://dexsuite.readthedocs.io/en/latest/how-to/new-arm.html) and
[end-effectors](https://dexsuite.readthedocs.io/en/latest/how-to/new-end-effector.html).

## Report a bug

Open an issue with the `ds.make` call, the full traceback, and the output of

```bash
python -c "import dexsuite, genesis, torch; print(dexsuite.__file__, genesis.__version__, torch.__version__, torch.cuda.is_available())"
```
