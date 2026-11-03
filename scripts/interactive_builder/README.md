# DexSuite Interactive Builder

This folder contains the repo-local interactive environment builder.

## Run (recommended)

From the repo root:

```bash
python -m scripts.interactive_builder
```

To skip teleop and just run the env:

```bash
python -m scripts.interactive_builder --input none
```

## Notes

- The builder is **TUI-first** (curses) and falls back to a simple prompt UI when
  curses is unavailable.
- `curses` is **not** a pip package on Linux/macOS. It’s a standard-library module.
  If you’re on Windows, the equivalent is `windows-curses`.
- The runner enforces **torch-only actions** (`env.step(action)` always receives
  a `torch.Tensor`).
- Pose controllers expose a backend selector in the builder. Use `mink` for the
  default solver path, or `genesis_internal` to use Genesis IK while keeping the
  same controller name.
- Teleop devices:
  - `keyboard`: requires `pynput`
  - `spacemouse`: requires `evdev` (Linux) or `hidapi` (macOS/Windows)
  - `vive_*`: requires `openvr`

To install optional device dependencies from PyPI:

```bash
pip install "dexsuite[devices]"
```
