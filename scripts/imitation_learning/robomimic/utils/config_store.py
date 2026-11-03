"""Load DexSuite presets for robomimic imitation-learning runs.

The preset files describe DexSuite-level choices, such as dataset identity,
policy family, and rollout cadence. They are intentionally thinner than
robomimic JSON configs, and are expanded into a concrete robomimic config at
runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
DATASETS_FILE = CONFIG_ROOT / "datasets.yaml"
POLICIES_DIR = CONFIG_ROOT / "policies"
ROLLOUTS_DIR = CONFIG_ROOT / "rollouts"
BASE_CONFIGS_DIR = CONFIG_ROOT / "base_configs"

_METADATA_KEYS = {
    "description",
    "algo_name",
    "obs_profile",
    "default_sensors",
    "no_images",
}
_POLICY_ALIASES = {
    "bc_mlp": "bc",
    "bc_mlp_lowdim": "bc",
    "bc_rnn": "bc",
    "bc_rnn_lowdim": "bc",
    "bc_transformer": "bc",
    "bc_transformer_lowdim": "bc",
    "bc_resnet": "bc",
    "bc_resnet_cameras": "bc",
    "diffusion_lowdim": "diffusion",
}


def load_dataset_registry() -> dict[str, dict[str, Any]]:
    """Load configured dataset aliases.

    Returns:
        Dataset registry keyed by command-line dataset name.

    Raises:
        FileNotFoundError: When the datasets YAML file is missing.
        ValueError: When the datasets YAML file has an invalid top-level shape.
    """
    registry = _read_yaml(DATASETS_FILE)
    if not isinstance(registry, dict):
        raise ValueError(f"Dataset registry must be a mapping: {DATASETS_FILE}")
    return {str(key): dict(value or {}) for key, value in registry.items()}


def load_policy_preset(name: str) -> dict[str, Any]:
    """Load a named policy preset.

    Args:
        name: Policy preset name, with or without the `.yaml` suffix.

    Returns:
        Policy preset dictionary.
    """
    return _load_named_yaml(POLICIES_DIR, _resolve_policy_alias(name))


def load_rollout_preset(name: str) -> dict[str, Any]:
    """Load a named rollout preset.

    Args:
        name: Rollout preset name, with or without the `.yaml` suffix.

    Returns:
        Rollout preset dictionary.
    """
    return _load_named_yaml(ROLLOUTS_DIR, name)


def list_policy_presets() -> list[str]:
    """List available policy preset names.

    Returns:
        Sorted preset names without file suffixes.
    """
    return _list_yaml_stems(POLICIES_DIR)


def list_rollout_presets() -> list[str]:
    """List available rollout preset names.

    Returns:
        Sorted preset names without file suffixes.
    """
    return _list_yaml_stems(ROLLOUTS_DIR)


def list_robomimic_base_configs() -> list[str]:
    """List checked-in robomimic JSON base config filenames.

    Returns:
        Sorted JSON filenames.
    """
    if not BASE_CONFIGS_DIR.is_dir():
        return []
    return sorted(path.name for path in BASE_CONFIGS_DIR.glob("*.json"))


def robomimic_base_config_path(name: str) -> Path | None:
    """Resolve a checked-in robomimic JSON base config by filename.

    Args:
        name: Config filename, with or without the `.json` suffix.

    Returns:
        Resolved config path, or None when no checked-in base config matches.
    """
    filename = name if name.endswith(".json") else f"{name}.json"
    path = BASE_CONFIGS_DIR / filename
    return path.resolve() if path.is_file() else None


def policy_algo_name(policy_preset: Mapping[str, Any] | None) -> str | None:
    """Return the robomimic algorithm requested by a policy preset.

    Args:
        policy_preset: Loaded policy preset, or None.

    Returns:
        Algorithm name, or None when no preset is active.
    """
    if policy_preset is None:
        return None
    algo_name = policy_preset.get("algo_name")
    return str(algo_name) if algo_name is not None else None


def policy_obs_profile(policy_preset: Mapping[str, Any] | None) -> str | None:
    """Return the dataset observation profile requested by a policy preset.

    Args:
        policy_preset: Loaded policy preset, or None.

    Returns:
        Observation profile name, or None when the preset leaves it unspecified.
    """
    if policy_preset is None:
        return None
    obs_profile = policy_preset.get("obs_profile")
    return str(obs_profile) if obs_profile is not None else None


def policy_no_images(policy_preset: Mapping[str, Any] | None) -> bool | None:
    """Return whether a policy preset should skip image downloads.

    Args:
        policy_preset: Loaded policy preset, or None.

    Returns:
        Boolean image-download preference, or None when unspecified.
    """
    if policy_preset is None or "no_images" not in policy_preset:
        return None
    return bool(policy_preset["no_images"])


def apply_preset_config(config: Any, preset: Mapping[str, Any] | None) -> None:
    """Apply a preset's robomimic config patch in place.

    Args:
        config: Robomimic config object to mutate.
        preset: Loaded preset with an optional `config` mapping.

    Raises:
        ValueError: When the preset config patch is not a mapping.
    """
    if preset is None:
        return
    patch = preset.get("config", {})
    if not isinstance(patch, Mapping):
        raise ValueError("Preset 'config' section must be a mapping.")
    _apply_mapping(config, patch)


def rollout_arg_defaults(preset: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return standalone rollout defaults from a rollout preset.

    Args:
        preset: Loaded rollout preset, or None.

    Returns:
        Dictionary of rollout command defaults.

    Raises:
        ValueError: When the preset rollout args are not a mapping.
    """
    if preset is None:
        return {}
    args = preset.get("rollout_args", {})
    if not isinstance(args, Mapping):
        raise ValueError("Preset 'rollout_args' section must be a mapping.")
    return dict(args)


def preset_summary(name: str, preset: Mapping[str, Any]) -> str:
    """Format one preset for list commands.

    Args:
        name: Preset name.
        preset: Loaded preset mapping.

    Returns:
        One-line preset description.
    """
    description = preset.get("description")
    return f"{name}: {description}" if description else name


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_policy_alias(name: str) -> str:
    stem = name[:-5] if name.endswith(".yaml") else name
    return _POLICY_ALIASES.get(stem, stem)


def _load_named_yaml(directory: Path, name: str) -> dict[str, Any]:
    path = _preset_path(directory, name)
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a mapping: {path}")
    return data


def _preset_path(directory: Path, name: str) -> Path:
    stem = name[:-5] if name.endswith(".yaml") else name
    path = directory / f"{stem}.yaml"
    if not path.is_file():
        known = ", ".join(_list_yaml_stems(directory)) or "<none>"
        raise FileNotFoundError(f"Unknown preset '{name}'. Known presets: {known}")
    return path


def _list_yaml_stems(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def _apply_mapping(target: Any, patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if key in _METADATA_KEYS:
            continue
        if isinstance(value, Mapping) and _has_child(target, key):
            child = _get_child(target, key)
            if _can_descend(child):
                _apply_mapping(child, value)
                continue
        _set_child(target, key, value)


def _has_child(target: Any, key: str) -> bool:
    if isinstance(target, dict):
        return key in target
    return hasattr(target, key)


def _get_child(target: Any, key: str) -> Any:
    if isinstance(target, dict):
        return target[key]
    return getattr(target, key)


def _set_child(target: Any, key: str, value: Any) -> None:
    if isinstance(target, dict):
        target[key] = value
    else:
        setattr(target, key, value)


def _can_descend(value: Any) -> bool:
    return (
        isinstance(value, dict) or hasattr(value, "keys") or hasattr(value, "__dict__")
    )
