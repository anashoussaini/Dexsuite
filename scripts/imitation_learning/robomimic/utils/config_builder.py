"""Apply DexSuite dataset metadata to robomimic configs.

The training entrypoint uses these helpers to merge dataset-local action and
observation metadata into robomimic config objects. Observation YAML files are
strictly resolved or generated from the dataset HDF5, so config generation does
not silently fall back to an unrelated base template.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install via: pip install pyyaml", file=sys.stderr)
    raise

try:
    from scripts.imitation_learning.robomimic.utils.obs_profiles import (
        resolve_observation_yaml,
    )
except ModuleNotFoundError:
    from utils.obs_profiles import resolve_observation_yaml


# --------------- I/O helpers ----------------


def read_json(p: str):
    with Path(p).expanduser().open() as f:
        return json.load(f)


def write_json(p: str, obj):
    path = Path(p).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def read_yaml(p: str):
    with Path(p).expanduser().open() as f:
        return yaml.safe_load(f)


def abspath(path: str) -> str:
    return str(Path(path).expanduser().resolve())


# --------------- dict helpers ----------------


def deep_set(d: dict, keys: list[str], value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def deep_get(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# --------------- dataset discovery ----------------


def discover_hdf5(dataset_root: str, override_hdf5: str | None) -> str:
    if override_hdf5:
        return abspath(override_hdf5)
    h5_dir = Path(dataset_root).expanduser() / "hdf5"
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"hdf5 folder not found under {dataset_root}")
    cands = [path for path in h5_dir.iterdir() if path.suffix == ".hdf5"]
    if not cands:
        raise FileNotFoundError(f"No .hdf5 files in {h5_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def discover_action_config(
    dataset_root: str,
    override_action_config: str | None,
) -> str:
    if override_action_config:
        p = Path(abspath(override_action_config))
    else:
        p = Path(dataset_root).expanduser() / "hdf5" / "action_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"action_config.json not found: {p}")
    return str(p)


def dataset_variant_name(dataset_root: str) -> str:
    # e.g., /.../datasets/lift_ur5_allegro/lift -> 'lift_ur5_allegro'
    return Path(dataset_root).expanduser().resolve().parent.name


def default_out_path(dataset_root: str, name: str | None) -> str:
    var = dataset_variant_name(dataset_root)
    base = name or f"{var}_base"
    return str(Path(dataset_root).expanduser() / "hdf5" / f"{base}.json")


# --------------- action_config -> train.* ----------------


def build_action_settings_from_json(
    action_cfg_json: str,
    action_normalization: str = "dataset",
):
    """Return robomimic action keys and executable action config.

    The dataset strategy trusts the converted dataset metadata, where actions
    already normalized by the environment are left untouched and raw continuous
    components use their dataset-specific robomimic normalization. Override
    strategies are intentionally coarse and apply the same normalization to each
    component.
    """
    ac = read_json(action_cfg_json)
    action_keys = ac.get("action_keys", [])
    if not action_keys:
        raise ValueError("action_config.json missing 'action_keys'.")

    action_config = _base_action_config(action_keys)
    if action_normalization == "dataset" and "robomimic_action_config" in ac:
        return action_keys, _validated_action_config(
            ac["robomimic_action_config"],
            action_keys,
        )

    seen = set()
    for seg in ac.get("segments", []):
        for part in seg.get("parts", []):
            key = part.get("robomimic_id")
            if key not in action_config:
                continue
            action_config[key] = _action_config_for_part(part, action_normalization)
            seen.add(key)

    missing = [key for key in action_keys if key not in seen]
    if missing:
        raise ValueError(f"action_config.json missing action parts for: {missing}")

    return action_keys, action_config


def _base_action_config(action_keys: list[str]) -> dict[str, dict[str, object]]:
    return {key: {"normalization": None, "rot_conversion": None} for key in action_keys}


def _validated_action_config(
    action_config: dict[str, dict[str, object]],
    action_keys: list[str],
) -> dict[str, dict[str, object]]:
    validated = _base_action_config(action_keys)
    for key in action_keys:
        if key not in action_config:
            raise ValueError(f"robomimic_action_config missing key: {key}")
        item = dict(action_config[key])
        item["normalization"] = _validate_normalization(item.get("normalization"))
        item.setdefault("rot_conversion", None)
        validated[key] = item
    return validated


def _action_config_for_part(
    part: dict[str, object],
    action_normalization: str,
) -> dict[str, object]:
    if action_normalization == "dataset":
        normalization = _dataset_normalization_for_part(part)
    elif action_normalization == "none":
        normalization = None
    elif action_normalization in ("min_max", "gaussian"):
        normalization = action_normalization
    else:
        raise ValueError(
            "action_normalization must be one of: dataset, none, min_max, gaussian",
        )

    config: dict[str, object] = {
        "normalization": normalization,
        "rot_conversion": part.get("rot_conversion"),
    }
    action_format = part.get("format")
    if action_format is not None:
        config["format"] = action_format
    convert_at_runtime = part.get("convert_at_runtime")
    if convert_at_runtime is not None:
        config["convert_at_runtime"] = convert_at_runtime
    return config


def _dataset_normalization_for_part(part: dict[str, object]) -> str | None:
    if part.get("normalized_in_env", False):
        return None
    return _validate_normalization(
        part.get("normalization", part.get("recommended_normalization")),
    )


def _validate_normalization(value: object) -> str | None:
    if value in (None, "none"):
        return None
    if value in ("min_max", "gaussian"):
        return str(value)
    raise ValueError(f"Unsupported action normalization: {value}")


# --------------- obs YAML -> observation.modalities.* ----------------


def resolve_obs_yaml_strict(
    dataset_root: str,
    obs_hint: str | None,
    obs_profile: str | None,
    h5_path: str,
) -> str:
    """Resolve the observation YAML path without falling back to base config.

    An explicit file or directory is used as-is after validation. A profile name
    generates or reuses the dataset-local `obs_<profile>.yaml` file from the HDF5
    observation keys.
    """
    candidates = ("obs_lowdim.yaml", "obs_cameras.yaml")

    if obs_hint:
        obs_path = Path(obs_hint).expanduser().resolve()
        if obs_path.is_dir():
            for name in candidates:
                p = obs_path / name
                if p.is_file():
                    return str(p)
            raise FileNotFoundError(
                f"No obs YAML found in directory: {obs_path}. Expected one of: {', '.join(candidates)}",
            )
        if not obs_path.is_file():
            raise FileNotFoundError(f"Obs YAML file not found: {obs_path}")
        return str(obs_path)

    if obs_profile:
        return str(
            resolve_observation_yaml(
                hdf5_path=h5_path,
                hdf5_dir=Path(dataset_root).expanduser() / "hdf5",
                obs_yaml=None,
                obs_profile=obs_profile,
            ),
        )

    raise ValueError(
        "You must provide --obs (file or directory) OR --obs-profile {lowdim,cameras}. "
        "No fallback to base observation config is allowed.",
    )


def apply_observation_yaml(config_obj: dict, obs_yaml_path: str) -> dict:
    """YAML schema:
    obs:
      low_dim: [...]
      rgb: [...]
      depth: [...]
    """
    spec = read_yaml(obs_yaml_path) or {}
    if "obs" not in spec:
        raise ValueError(f"Obs YAML missing top-level 'obs' key: {obs_yaml_path}")

    obs = spec["obs"]
    low_dim = obs.get("low_dim", [])
    rgb = obs.get("rgb", [])
    depth = obs.get("depth", [])

    deep_set(config_obj, ["observation", "modalities", "obs", "low_dim"], list(low_dim))
    deep_set(config_obj, ["observation", "modalities", "obs", "rgb"], list(rgb))
    deep_set(config_obj, ["observation", "modalities", "obs", "depth"], list(depth))
    return config_obj


# --------------- optional strict verification ----------------


def _list_obs_keys_in_h5(h5_path: str):
    import h5py

    with h5py.File(h5_path, "r") as f:
        demos = list(f["data"].keys())
        if not demos:
            return set()
        demo = demos[0]
        return set(f[f"data/{demo}/obs"].keys())


def _list_action_keys_in_h5(h5_path: str):
    import h5py

    present = set()
    with h5py.File(h5_path, "r") as f:
        demos = list(f["data"].keys())
        if not demos:
            return present
        demo = demos[0]
        grp = f[f"data/{demo}"]
        # top-level datasets under data/<demo> (e.g., "actions", "rewards", ...)
        for k in grp:
            present.add(str(k))  # e.g., "actions", "obs", "action"
        # descend into data/<demo>/action/* and return as "action/<name>"
        if "action" in grp and isinstance(grp["action"], h5py.Group):
            for name in grp["action"]:
                present.add(f"action/{name}")
    return present


def enforce_obs_keys(h5_path: str, obs_yaml_path: str, strict: bool):
    try:
        import h5py  # noqa: F401
    except ImportError:
        if strict:
            raise RuntimeError(
                "h5py not installed but --strict-obs-keys was set.",
            ) from None
        print("[WARN] h5py not installed; skipping obs key verification.")
        return
    wanted = set()
    spec = read_yaml(obs_yaml_path) or {}
    for group in ("low_dim", "rgb", "depth"):
        for k in spec.get("obs", {}).get(group, []) or []:
            wanted.add(k)
    if not wanted:
        if strict:
            raise ValueError(f"No obs keys listed in {obs_yaml_path} (strict mode).")
        print(f"[WARN] No obs keys listed in {obs_yaml_path}.")
        return
    present = _list_obs_keys_in_h5(h5_path)
    missing = sorted(k for k in wanted if k not in present)
    if missing:
        msg = f"Obs keys missing in {Path(h5_path).name}: {missing}"
        if strict:
            raise ValueError(msg)
        print("[WARN]", msg)


def enforce_action_keys(h5_path: str, action_keys: list[str] | None, strict: bool):
    if not action_keys:
        return
    try:
        import h5py  # noqa: F401
    except ImportError:
        if strict:
            raise RuntimeError(
                "h5py not installed but --strict-action-keys was set.",
            ) from None
        print("[WARN] h5py not installed; skipping action key verification.")
        return

    present = _list_action_keys_in_h5(h5_path)
    missing = sorted(k for k in action_keys if k not in present)
    if missing:
        msg = (
            f"Action keys missing in {Path(h5_path).name}: {missing}\n"
            f"  present (sample): {sorted(present)[:20]}"
        )
        if strict:
            raise ValueError(msg)
        print("[WARN]", msg)
