"""Expose the public DexSuite package API.

DexSuite imports bundled models and environments at package import time so
registry decorators populate the global registries. Device and environment-count
state is process-local and shared by the public helpers re-exported here.

Subpackages:
    controllers: Controller factories and built-in control modes.
    core: Robot components, registries, camera systems, and robot wrappers.
    environments: Registered simulation task environments.
    models: Robot, arena, object, and texture model assets.
    options: Dataclass-style configuration objects for environment creation.
    utils: Shared utility helpers for tensors, assets, logging, and display.
"""

from __future__ import annotations

import importlib
import pkgutil

__version__ = "0.1.1"
# ----------------------------------------------------------------------
# public surface (core functions)
# ----------------------------------------------------------------------
from .api import make  # re-export

# ----------------------------------------------------------------------
# public surface (configuration options)
# These are imported from the subpackage to be available directly under dexsuite.
# ----------------------------------------------------------------------
from .options import (
    AABBOptions,
    ArmOptions,
    CamerasOptions,
    ControllerOptions,
    DynamicCamOptions,
    EnvOptions,
    LayoutOptions,
    PoseOptions,
    RobotOptions,
    SimOptions,
    StaticCamOptions,
)
from .utils import get_device, set_device

__all__ = [
    "AABBOptions",
    "ArmOptions",
    "CamerasOptions",
    "ControllerOptions",
    "DynamicCamOptions",
    "EnvOptions",
    "LayoutOptions",
    "PoseOptions",
    "RobotOptions",
    "SimOptions",
    "StaticCamOptions",
    "get_device",
    "make",
    "set_device",
]


# ----------------------------------------------------------------------
# auto-import sub-packages so that the registries are populated
# ----------------------------------------------------------------------
def _autoimport(pkg_name: str) -> None:
    """Import every submodule under a package so decorators can register classes.

    Args:
        pkg_name: Fully qualified package name to walk.
    """
    pkg = importlib.import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)


_autoimport(__name__ + ".models")
_autoimport(__name__ + ".environments")
