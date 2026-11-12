"""Install robomimic 0.5.0 for DexSuite imitation learning.

robomimic 0.5.0 is not on PyPI (PyPI has 0.3.0, which these recipes do not run on), and its own dependency pins
(huggingface_hub 0.23, transformers 4.41, diffusers 0.11) would downgrade what LeRobot and the diffusion policy need.
So it is installed from its v0.5.0 tag without its dependencies; the ones it really uses come with dexsuite's
"il" extra. From the repository root:

    pip install -e ".[il]"
    python -m scripts.imitation_learning.robomimic.install

Works in pip and uv environments. Run it again any time; it only reinstalls robomimic.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys

ROBOMIMIC = "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0"
VERSION = "0.5.0"


def installed_version() -> str | None:
    """The installed robomimic version, or None."""
    try:
        return importlib.metadata.version("robomimic")
    except importlib.metadata.PackageNotFoundError:
        return None


def install_command() -> list[str]:
    """The install command: pip in this interpreter if it has pip, else uv pointed at this interpreter."""
    has_pip = (
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"], capture_output=True, check=False
        ).returncode
        == 0
    )
    if has_pip:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            ROBOMIMIC,
        ]
    uv = shutil.which("uv")
    if uv:
        return [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            ROBOMIMIC,
        ]
    raise SystemExit(
        "Neither pip nor uv is available for this Python; install one of them first."
    )


def main() -> None:
    """Install robomimic 0.5.0 without its pins, then check it and the dependencies the recipes import."""
    cmd = install_command()
    print("[install] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)  # noqa: S603 - a fixed command: this interpreter's pip or uv, and a pinned tag
    importlib.invalidate_caches()
    version = installed_version()
    if version != VERSION:
        raise SystemExit(f"[install] expected robomimic {VERSION}, found {version}")
    needed = (
        "h5py",
        "tensorboardX",
        "transformers",
        "diffusers",
        "imageio",
        "matplotlib",
        "termcolor",
        "psutil",
    )
    missing = [module for module in needed if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit(
            f'[install] robomimic is in, but {", ".join(missing)} are missing: pip install -e ".[il]"'
        )
    print(
        f"[install] robomimic {version} ready for scripts/imitation_learning/robomimic/train.py",
        flush=True,
    )


if __name__ == "__main__":
    main()
