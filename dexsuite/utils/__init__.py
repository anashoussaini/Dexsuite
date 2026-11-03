"""Expose shared utility helpers for DexSuite.

This package keeps common helpers in one place and re-exports a small, stable surface area for device globals, randomizers, and math utilities.

Submodules:
    aabb: Torch axis-aligned bounding box utilities.
    action_utils: Action layout and dispatch helpers.
    globals: Process-local runtime defaults.
    lerobot_logger: LeRobot dataset logging integration.
    models_utils: Asset path resolution helpers.
    options_utils: YAML configuration loading helpers.
    orientation_utils: Quaternion and Euler angle conversion helpers.
    randomizers: Position and yaw sampling helpers.
    robot_utils: Genesis robot construction helpers.
    space_utils: Gymnasium space construction helpers.
    status: Console status display helpers.
    tactile_utils: Rigid tactile pad observation helpers.
    workspace_utils: Workspace AABB utilities.
"""

from __future__ import annotations

from .globals import get_device, get_n_envs, set_device, set_n_envs
from .models_utils import get_object_path, get_texture_paths
from .options_utils import load_component_yaml
from .orientation_utils import quat_to_rpy, rpy_to_quat
from .randomizers import (
    ThreeDPosRandomizer,
    TwoDPosRandomizer,
    YawRandomizer,
    sampler_in_aabb,
)
from .space_utils import dict_to_space


def get_robot_morph(*args, **kwargs):
    """Lazy import wrapper for dexsuite.utils.robot_utils.get_robot_morph.

        Importing dexsuite.utils should stay lightweight; Genesis is only needed when
        robot morphs are actually constructed.

    Args:
        args: Positional arguments forwarded to the wrapped helper.
        kwargs: Keyword arguments forwarded to the wrapped helper.
    """
    from .robot_utils import get_robot_morph as _get_robot_morph

    return _get_robot_morph(*args, **kwargs)


__all__ = [
    "ThreeDPosRandomizer",
    "TwoDPosRandomizer",
    "YawRandomizer",
    "dict_to_space",
    "get_device",
    "get_n_envs",
    "get_object_path",
    "get_robot_morph",
    "get_texture_paths",
    "load_component_yaml",
    "quat_to_rpy",
    "rpy_to_quat",
    "sampler_in_aabb",
    "set_device",
    "set_n_envs",
]
