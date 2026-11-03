"""Infer robomimic observation profiles from converted DexSuite datasets.

Observation profiles are dataset-local YAML files because the available keys and
image dimensions come from the recorded environment. Profiles classify each HDF5
observation key into robomimic modalities and keep proprioceptive robot keys
separate from environment state metadata.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

ObservationProfileName = Literal["lowdim", "cameras"]
LowDimSelection = Literal["all", "proprio", "state"]

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_OBSERVATIONS_FILE = _CONFIG_ROOT / "observations.yaml"
_IMAGE_CHANNEL_COUNTS = {1, 3, 4}
_DEPTH_NAME_TOKENS = ("depth", "distance", "range")
_RGB_NAME_TOKENS = ("rgb", "image", "camera", "cam", "front", "wrist", "overhead")
_PROPRIO_NAME_TOKENS = (
    "eef",
    "tcp",
    "joint",
    "qpos",
    "qvel",
    "qtorque",
    "manipulator",
    "gripper",
    "robot",
)
_STATE_ROBOT_KEYS = {
    "state_gripper_qpos",
    "state_manipulator_eef_pos",
}
_STATE_LOW_DIM_ORDER = (
    "state_manipulator_eef_pos",
    "state_gripper_qpos",
    "state_other_cube_pos",
    "state_other_target_pos",
)
_STATE_OTHER_EXCLUDE_KEYS = {
    "state_other_action",
    "state_other_last_action",
}


@dataclass(frozen=True)
class ObservationProfile:
    """Group HDF5 observation keys by robomimic modality.

    Attributes:
        low_dim: Keys consumed by low-dimensional encoders.
        rgb: Keys consumed by image encoders.
        depth: Keys consumed by depth encoders.
        proprio: Low-dimensional keys available from robot sensors.
        state: Low-dimensional keys that describe environment or simulator state.
    """

    low_dim: tuple[str, ...]
    rgb: tuple[str, ...]
    depth: tuple[str, ...]
    proprio: tuple[str, ...] = ()
    state: tuple[str, ...] = ()


def default_camera_names() -> tuple[str, ...]:
    """Return the default camera names used for camera policies."""
    cameras = _observation_preset_mapping().get("cameras", {})
    if not isinstance(cameras, dict):
        raise TypeError(f"'cameras' must be a mapping in {_OBSERVATIONS_FILE}")
    return _string_tuple(cameras.get("default_names", ()))


def task_low_dim_keys(task: str | None) -> tuple[str, ...]:
    """Return the configured low-dimensional policy keys for a task."""
    if task is None:
        return ()
    tasks = _observation_preset_mapping().get("tasks", {})
    if not isinstance(tasks, dict):
        raise TypeError(f"'tasks' must be a mapping in {_OBSERVATIONS_FILE}")
    task_spec = tasks.get(str(task)) or tasks.get(_safe_name(task))
    if task_spec is None:
        return ()
    if not isinstance(task_spec, dict):
        raise TypeError(f"Task observation preset must be a mapping: {task}")
    return _string_tuple(task_spec.get("low_dim", ()))


def hdf5_has_visual_observations(hdf5_path: str | Path) -> bool:
    """Return whether the first HDF5 demo contains image or depth observations."""
    import h5py

    hdf5_path = Path(hdf5_path).expanduser().resolve()
    if not hdf5_path.is_file():
        return False

    with h5py.File(hdf5_path, "r") as file:
        data = file.get("data")
        if data is None or len(data) == 0:
            return False
        demo_key = sorted(data.keys())[0]
        obs_group = file.get(f"data/{demo_key}/obs")
        if obs_group is None:
            return False
        for key in obs_group:
            dataset = obs_group[key]
            modality = _classify_observation_key(
                str(key),
                tuple(dataset.shape[1:]),
                np.dtype(dataset.dtype),
            )
            if modality in {"rgb", "depth"}:
                return True
    return False


def infer_observation_profile(
    hdf5_path: str | Path,
    profile_name: ObservationProfileName,
    exclude_keys: Iterable[str] | None = None,
    camera_names: Iterable[str] | None = None,
) -> ObservationProfile:
    """Build an observation profile from the first demo in an HDF5 dataset.

    The lowdim profile excludes image-like arrays so accidental camera keys do
    not get wired into low-dimensional encoders. The cameras profile keeps
    low-dimensional keys and adds only the requested visual observations.

    Args:
        hdf5_path: Converted robomimic HDF5 dataset to inspect.
        profile_name: Profile to generate.
        exclude_keys: Observation keys to omit from the generated profile.
        camera_names: Camera name fragments to keep for visual profiles. Defaults
            to None.

    Returns:
        Observation keys grouped by robomimic modality.

    Raises:
        ValueError: When the dataset has no demos, no observations, or an unknown
            profile is requested.
    """
    import h5py

    if profile_name not in ("lowdim", "cameras"):
        raise ValueError(f"Unknown obs profile: {profile_name}")

    hdf5_path = Path(hdf5_path).expanduser().resolve()
    excluded = {str(key) for key in exclude_keys or ()}
    selected_cameras = {str(name).lower() for name in camera_names or ()}
    low_dim: list[str] = []
    rgb: list[str] = []
    depth: list[str] = []
    proprio: list[str] = []
    state: list[str] = []

    with h5py.File(hdf5_path, "r") as file:
        data = file.get("data")
        if data is None or len(data) == 0:
            raise ValueError(f"HDF5 dataset has no demos: {hdf5_path}")

        demo_key = sorted(data.keys())[0]
        obs_group = file.get(f"data/{demo_key}/obs")
        if obs_group is None or len(obs_group) == 0:
            raise ValueError(f"HDF5 demo has no obs group: data/{demo_key}/obs")

        for key in sorted(obs_group.keys()):
            key_name = str(key)
            if key_name in excluded:
                continue
            dataset = obs_group[key]
            modality = _classify_observation_key(
                key_name,
                tuple(dataset.shape[1:]),
                np.dtype(dataset.dtype),
            )
            _append_profile_key(
                key_name,
                modality,
                selected_cameras,
                low_dim,
                rgb,
                depth,
                proprio,
                state,
            )

    profile = ObservationProfile(
        tuple(low_dim),
        tuple(rgb),
        tuple(depth),
        proprio=tuple(proprio),
        state=tuple(state),
    )
    if profile_name == "lowdim":
        return ObservationProfile(
            profile.low_dim,
            (),
            (),
            proprio=profile.proprio,
            state=profile.state,
        )
    if not profile.rgb and not profile.depth:
        raise ValueError(
            f"Obs profile 'cameras' was requested, but {hdf5_path} contains no "
            "image or depth observation keys. Convert without --no-images or use "
            "--sensors state / --sensors proprio.",
        )
    return profile


def write_observation_profile_yaml(
    profile: ObservationProfile,
    out_path: str | Path,
    low_dim_selection: LowDimSelection = "all",
    task: str | None = None,
) -> Path:
    """Write an observation profile in the schema consumed by robomimic configs.

    Args:
        profile: Observation profile to serialize.
        out_path: Destination YAML path.
        low_dim_selection: Low-dimensional group to expose to robomimic.
            Defaults to all.
        task: Optional DexSuite task name used to select a task-specific
            low-dimensional policy profile.

    Returns:
        Resolved path to the written YAML file.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    low_dim_keys = _selected_low_dim_keys(
        profile,
        low_dim_selection,
        task=task,
    )
    with out_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "obs": {
                    "low_dim": list(low_dim_keys),
                    "rgb": list(profile.rgb),
                    "depth": list(profile.depth),
                },
                "groups": {
                    "proprio": list(profile.proprio),
                    "state": list(profile.state),
                },
            },
            file,
            sort_keys=False,
        )
    return out_path


def resolve_observation_yaml(
    *,
    hdf5_path: str | Path,
    hdf5_dir: str | Path,
    obs_yaml: str | Path | None,
    obs_profile: ObservationProfileName | None,
    exclude_keys: Iterable[str] | None = None,
    camera_names: Iterable[str] | None = None,
    low_dim_selection: LowDimSelection = "all",
    task: str | None = None,
) -> Path | None:
    """Resolve or generate the dataset-local observation YAML file.

    Explicit YAML files are copied beside the HDF5 dataset so generated configs
    are self-contained. Profile names generate deterministic files from the HDF5
    observation shapes when the file is missing.

    Args:
        hdf5_path: Converted robomimic HDF5 dataset to inspect when generating.
        hdf5_dir: Directory that stores dataset-local HDF5 metadata.
        obs_yaml: Explicit observation YAML path, or None to use a profile.
        obs_profile: Named observation profile to generate or reuse. Defaults to
            None.
        exclude_keys: Observation keys to omit from the resolved YAML.
        camera_names: Camera name fragments to keep for visual profiles. Defaults
            to None.
        low_dim_selection: Low-dimensional group to expose to robomimic.
            Defaults to all.
        task: Optional DexSuite task name used to select a task-specific
            low-dimensional policy profile.

    Returns:
        Resolved observation YAML path, or None when no profile or explicit YAML
        is requested.

    Raises:
        FileNotFoundError: When an explicit observation YAML path does not exist.
    """
    hdf5_dir = Path(hdf5_dir).expanduser().resolve()
    excluded = {str(key) for key in exclude_keys or ()}
    if obs_yaml is not None:
        source = Path(obs_yaml).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Obs YAML not found: {source}")
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        dest_name = source.name
        if obs_profile is not None:
            dest_name = _observation_yaml_name(
                obs_profile,
                dataset_name=Path(hdf5_path).stem,
                camera_names=camera_names,
                low_dim_selection=low_dim_selection,
            )
        dest = hdf5_dir / dest_name
        if source != dest:
            shutil.copyfile(source, dest)
        if excluded:
            _filter_observation_yaml(dest, excluded)
        return dest

    if obs_profile is None:
        return None

    out_path = hdf5_dir / _observation_yaml_name(
        obs_profile,
        dataset_name=Path(hdf5_path).stem,
        camera_names=camera_names,
        low_dim_selection=low_dim_selection,
    )
    profile = infer_observation_profile(
        hdf5_path,
        obs_profile,
        exclude_keys=excluded,
        camera_names=camera_names,
    )
    return write_observation_profile_yaml(
        profile,
        out_path,
        low_dim_selection,
        task=task,
    )


def _filter_observation_yaml(path: Path, exclude_keys: set[str]) -> None:
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    obs = spec.setdefault("obs", {})
    for group in ("low_dim", "rgb", "depth"):
        obs[group] = [key for key in obs.get(group, []) if key not in exclude_keys]
    groups = spec.setdefault("groups", {})
    for group in ("proprio", "state"):
        groups[group] = [
            key for key in groups.get(group, []) if key not in exclude_keys
        ]
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _observation_yaml_name(
    obs_profile: ObservationProfileName,
    *,
    dataset_name: str | None,
    camera_names: Iterable[str] | None,
    low_dim_selection: LowDimSelection,
) -> str:
    parts = []
    if dataset_name:
        parts.append(_safe_name(dataset_name))
    parts.extend(["obs", obs_profile])
    if low_dim_selection != "all":
        parts.append(low_dim_selection)
    if obs_profile == "cameras":
        parts.extend(_safe_name(name) for name in camera_names or ())
    return "_".join(parts) + ".yaml"


def _safe_name(value: object) -> str:
    return str(value).strip().lower().replace("/", "_").replace(" ", "_")


def _selected_low_dim_keys(
    profile: ObservationProfile,
    selection: LowDimSelection,
    *,
    task: str | None = None,
) -> tuple[str, ...]:
    if selection == "all":
        return profile.low_dim
    if selection == "proprio":
        task_keys = _available_task_proprio_keys(profile, task)
        if task_keys:
            return task_keys
        return profile.proprio or profile.low_dim
    if selection == "state":
        task_keys = _available_task_low_dim_keys(profile, task)
        if task_keys:
            return task_keys
        selected = tuple(key for key in _STATE_LOW_DIM_ORDER if key in profile.low_dim)
        if selected:
            return selected
        extras = tuple(
            key
            for key in profile.low_dim
            if (
                key in _STATE_ROBOT_KEYS
                or (
                    key.startswith("state_other_")
                    and key not in _STATE_OTHER_EXCLUDE_KEYS
                )
            )
            and key not in selected
        )
        return selected + extras
    raise ValueError(f"Unknown low-dimensional selection: {selection}")


def _available_task_low_dim_keys(
    profile: ObservationProfile,
    task: str | None,
) -> tuple[str, ...]:
    configured = task_low_dim_keys(task)
    if not configured:
        return ()
    available = set(profile.low_dim)
    return tuple(key for key in configured if key in available)


def _available_task_proprio_keys(
    profile: ObservationProfile,
    task: str | None,
) -> tuple[str, ...]:
    return tuple(
        key
        for key in _available_task_low_dim_keys(profile, task)
        if not key.startswith("state_other_")
    )


@lru_cache(maxsize=1)
def _observation_preset_mapping() -> dict[str, Any]:
    if not _OBSERVATIONS_FILE.is_file():
        return {}
    with _OBSERVATIONS_FILE.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Observation presets must be a mapping: {_OBSERVATIONS_FILE}")
    return data


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected a list of strings in {_OBSERVATIONS_FILE}")
    return tuple(str(item) for item in value)


def _append_profile_key(
    key_name: str,
    modality: str,
    selected_cameras: set[str],
    low_dim: list[str],
    rgb: list[str],
    depth: list[str],
    proprio: list[str],
    state: list[str],
) -> None:
    if modality == "depth":
        if _camera_selected(key_name, selected_cameras):
            depth.append(key_name)
        return
    if modality == "rgb":
        if _camera_selected(key_name, selected_cameras):
            rgb.append(key_name)
        return

    low_dim.append(key_name)
    if _is_proprio_key(key_name):
        proprio.append(key_name)
    else:
        state.append(key_name)


def _is_proprio_key(key: str) -> bool:
    lower_key = key.lower()
    return any(token in lower_key for token in _PROPRIO_NAME_TOKENS)


def _camera_selected(key: str, camera_names: set[str]) -> bool:
    if not camera_names:
        return True
    lower_key = key.lower()
    return any(camera_name in lower_key for camera_name in camera_names)


def _classify_observation_key(
    key: str,
    per_step_shape: tuple[int, ...],
    dtype: np.dtype,
) -> str:
    lower_key = key.lower()
    depth_named = any(token in lower_key for token in _DEPTH_NAME_TOKENS)
    rgb_named = any(token in lower_key for token in _RGB_NAME_TOKENS)
    image_like = _is_channel_image_shape(per_step_shape) or (
        len(per_step_shape) == 2 and (depth_named or rgb_named)
    )

    if depth_named and (image_like or len(per_step_shape) >= 2):
        return "depth"
    if image_like and (rgb_named or np.issubdtype(dtype, np.integer)):
        return "rgb"
    if image_like:
        return "rgb"
    return "low_dim"


def _is_channel_image_shape(per_step_shape: tuple[int, ...]) -> bool:
    if len(per_step_shape) == 3 and per_step_shape[-1] in _IMAGE_CHANNEL_COUNTS:
        return True
    return len(per_step_shape) == 3 and per_step_shape[0] in _IMAGE_CHANNEL_COUNTS
