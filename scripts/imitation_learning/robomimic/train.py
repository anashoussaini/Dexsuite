#!/usr/bin/env python3
"""DexSuite robomimic training entrypoint.

This wraps the upstream robomimic train script with dataset preparation:
  1. resolve a logical dataset name through config/datasets.yaml,
  2. download / reuse a LeRobot dataset,
  3. convert it to robomimic HDF5 when needed,
  4. merge policy and rollout presets into a robomimic config,
  5. generate exact action and observation settings from dataset metadata,
  6. call robomimic's standard train loop, including DexSuite rollouts.

The managed layout is:
  datasets/<task>/lerobot/<repo-or-run>/
  datasets/<task>/hdf5/<dataset>.hdf5
  datasets/<task>/hdf5/action_config.json
  datasets/<task>/hdf5/obs_*.yaml
  datasets/<task>/configs/<experiment>.json
  runs/imitation_learning/robomimic/<task>/<robomimic output tree>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _require_robomimic() -> None:
    """robomimic 0.5.0 is needed and is not on PyPI: say how to get it instead of failing deep in an import."""
    import importlib.metadata

    how = 'pip install -e ".[il]" && python -m scripts.imitation_learning.robomimic.install'
    try:
        version = importlib.metadata.version("robomimic")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"robomimic is not installed. From the repository root: {how}") from None
    if not version.startswith("0.5."):
        raise SystemExit(f"robomimic {version} is installed; these recipes need 0.5.0 (PyPI only has 0.3.0): {how}")


_require_robomimic()

import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from robomimic.config import config_factory  # noqa: E402


try:
    from scripts.imitation_learning.robomimic.custom_algorithms import (
        register_custom_algorithms,
    )
    from scripts.imitation_learning.robomimic.utils import config_store
    from scripts.imitation_learning.robomimic.utils.obs_profiles import (
        default_camera_names,
        hdf5_has_visual_observations,
        resolve_observation_yaml,
    )
    from scripts.imitation_learning.robomimic.utils.rollout_env import (
        ensure_dexsuite_rollout_patches,
        obs_keys_from_config,
    )
except ModuleNotFoundError:
    from custom_algorithms import register_custom_algorithms
    from utils import config_store
    from utils.obs_profiles import (
        default_camera_names,
        hdf5_has_visual_observations,
        resolve_observation_yaml,
    )
    from utils.rollout_env import ensure_dexsuite_rollout_patches, obs_keys_from_config


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str | None = None
    task: str | None = None
    obs_yaml: str | None = None
    obs_profile: str | None = None
    camera_names: tuple[str, ...] = ()
    obs_exclude_keys: tuple[str, ...] = ()
    h5_basename: str | None = None
    action_normalization: str = "dataset"
    revision: str | None = None
    no_images: bool | None = None
    default_sensors: str | None = None


@dataclass(frozen=True)
class PreparedDataset:
    key: str
    task: str
    task_root: Path
    raw_root: Path | None
    hdf5_dir: Path
    hdf5_path: Path
    action_config_path: Path
    action_normalization: str
    obs_yaml_path: Path | None
    configs_dir: Path
    runs_dir: Path


@lru_cache(maxsize=1)
def _converter_module():
    try:
        from scripts.imitation_learning.robomimic.utils import converter
    except ModuleNotFoundError:
        from utils import converter

    return converter


@lru_cache(maxsize=1)
def _config_module():
    try:
        from scripts.imitation_learning.robomimic.utils import config_builder
    except ModuleNotFoundError:
        from utils import config_builder

    return config_builder


@lru_cache(maxsize=1)
def _dataset_registry() -> dict[str, dict[str, Any]]:
    return config_store.load_dataset_registry()


def _config_file_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix.lower() == ".json":
        base_config = config_store.robomimic_base_config_path(path.name)
        if base_config is not None:
            return base_config
        return path.resolve()
    return None


def _config_is_preset(args: argparse.Namespace) -> bool:
    return args.config is not None and _config_file_path(args.config) is None


def _dataset_file_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file() or path.suffix.lower() in {".h5", ".hdf5"}:
        return path.resolve()
    return None


def _load_policy_preset(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.policy_preset is not None:
        return config_store.load_policy_preset(args.policy_preset)
    if _config_is_preset(args):
        args.policy_preset = args.config
        return config_store.load_policy_preset(args.config)
    if args.config is None and args.algo is None:
        args.policy_preset = "bc"
        return config_store.load_policy_preset(args.policy_preset)
    return None


def _load_rollout_preset(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.rollout_preset is not None:
        return config_store.load_rollout_preset(args.rollout_preset)
    if args.config is None or _config_is_preset(args):
        args.rollout_preset = "standard"
        return config_store.load_rollout_preset(args.rollout_preset)
    return None


def _robomimic_train(config, *, device, resume: bool) -> None:
    if config.experiment.rollout.enabled:
        ensure_dexsuite_rollout_patches(obs_keys_from_config(config))

    import robomimic.scripts.train as train_module

    with (
        _dataloader_cpu_generator_patch(),
        _diffusion_policy_ema_patch(),
        _skip_visual_obs_normalization_patch(),
        _checkpoint_saving_patch(
            enabled=bool(config.experiment.save.enabled),
            train_module=train_module,
        ),
    ):
        train_module.train(config, device=device, resume=resume)


@contextmanager
def _dataloader_cpu_generator_patch():
    import torch
    from torch.utils.data import DataLoader

    original_generator = torch.Generator
    original_iter = DataLoader.__iter__

    def generator(*args, **kwargs):
        if kwargs.get("device") is not None:
            kwargs["device"] = "cpu"
        elif args and str(args[0]).startswith("cuda"):
            args = ("cpu", *args[1:])
        return original_generator(*args, **kwargs)

    def iter_with_generator_device(self):
        generator = getattr(self, "generator", None)
        # Genesis sets the global default device to cuda once a rollout env is
        # created; samplers hold cpu generators, so pin iteration to cpu.
        generator_device = getattr(generator, "device", None) or torch.device("cpu")
        with torch.device(generator_device):
            iterator = original_iter(self)
        return _DeviceBoundDataLoaderIterator(iterator, generator_device, torch)

    torch.Generator = generator
    DataLoader.__iter__ = iter_with_generator_device
    try:
        yield
    finally:
        torch.Generator = original_generator
        DataLoader.__iter__ = original_iter


class _DeviceBoundDataLoaderIterator:
    def __init__(self, iterator, device, torch_module):
        self._iterator = iterator
        self._device = device
        self._torch = torch_module

    def __iter__(self):
        return self

    def __next__(self):
        with self._torch.device(self._device):
            return next(self._iterator)

    def __getattr__(self, name):
        return getattr(self._iterator, name)


@contextmanager
def _diffusion_policy_ema_patch():
    """Bridge robomimic's diffusion policy to the current diffusers EMAModel API.

    robomimic's diffusion_policy.py (unreleased GitHub master, not the PyPI 0.3.0
    build) still targets the pre-0.31 diffusers EMA API: ``EMAModel(model=nets,
    power=...)``, ``ema.step(nets)``, and ``ema.averaged_model``. diffusers>=0.31
    replaced that with a parameters-based EMAModel that has no ``model=`` kwarg,
    no ``.averaged_model``, and a ``.step()`` that expects raw parameters, which
    raises ``EMAModel.__init__() missing 1 required positional argument:
    'parameters'`` when robomimic constructs it. This swaps in an adapter that
    keeps robomimic's expected attribute surface while delegating to the real
    diffusers EMAModel underneath.
    """
    try:
        import robomimic.algo.diffusion_policy as diffusion_policy_module
    except ImportError:
        yield
        return

    original_ema_model = diffusion_policy_module.EMAModel
    diffusion_policy_module.EMAModel = _LegacyEMAModelAdapter
    try:
        yield
    finally:
        diffusion_policy_module.EMAModel = original_ema_model


class _LegacyEMAModelAdapter:
    """Old-API-shaped EMA wrapper (``model=``, ``.step(model)``, ``.averaged_model``)."""

    def __init__(self, model, power=0.75, **kwargs):
        import copy

        from diffusers.training_utils import EMAModel as _DiffusersEMAModel

        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.requires_grad_(False)
        self._ema = _DiffusersEMAModel(list(model.parameters()), power=power, **kwargs)

    def step(self, model):
        self._ema.step(model.parameters())
        self._ema.copy_to(self.averaged_model.parameters())

@contextmanager
def _obs_normalize_debug_patch():
    """Print the exact shape mismatch for every failing modality in
    ObsUtils.normalize_dict before letting the original assertion fire."""
    import robomimic.utils.obs_utils as ObsUtils

    original_normalize_dict = ObsUtils.normalize_dict

    def debug_normalize_dict(dict_, normalization_stats):
        for m in dict_:
            stats = normalization_stats.get(m) if normalization_stats else None
            if stats is None:
                continue
            offset = stats["offset"]
            o_num_dims = len(offset.shape)
            value = dict_[m]
            actual_shape = getattr(value, "shape", None)
            trailing = tuple(actual_shape[-o_num_dims:]) if actual_shape is not None else None
            expected = tuple(offset.shape)
            if trailing != expected:
                print(
                    f"[obs-debug] MISMATCH key={m!r} "
                    f"full_obs_shape={tuple(actual_shape) if actual_shape is not None else None} "
                    f"trailing_dims_compared={trailing} "
                    f"expected_offset_shape={expected} "
                    f"(o_num_dims={o_num_dims}) "
                    f"dtype={getattr(value, 'dtype', None)}"
                )
        return original_normalize_dict(dict_, normalization_stats)

    ObsUtils.normalize_dict = debug_normalize_dict
    try:
        yield
    finally:
        ObsUtils.normalize_dict = original_normalize_dict

@contextmanager
def _skip_visual_obs_normalization_patch():
    """robomimic normally never mean/std-normalizes rgb/depth observations
    via ObsUtils.normalize_dict -- process_obs's built-in /255 scaling is the
    only normalization images get. But algo.py's rollout path
    (_prepare_observation) calls normalize_dict BEFORE process_obs, while
    dataset.py's normalize_obs() computes the hdf5 stats AFTER calling
    process_obs_dict (i.e. on already channel-first, /255-scaled data). That
    ordering mismatch means normalize_dict's shape check can never
    reconcile raw live (H, W, C) rollout observations against the
    post-process_obs (C, H, W) stats -- transposing the live data to match
    normalize_dict just breaks the immediately-following process_obs call
    instead (which needs raw channel-last input to do its own transpose).
 
    Fix: exclude rgb/depth keys from what normalize_dict ever sees, so they
    pass through completely untouched into process_obs, which then handles
    them exactly as it does for training batches. Low-dim keys are
    unaffected and normalized as before.
    """
    import robomimic.utils.obs_utils as ObsUtils
 
    original_normalize_dict = ObsUtils.normalize_dict
 
    def patched_normalize_dict(dict_, normalization_stats):
        visual_keys = {
            k for k in dict_
            if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb")
            or ObsUtils.key_is_obs_modality(key=k, obs_modality="depth")
        }
        if not visual_keys:
            return original_normalize_dict(dict_, normalization_stats)
 
        non_visual = {k: v for k, v in dict_.items() if k not in visual_keys}
        filtered_stats = {
            k: v for k, v in (normalization_stats or {}).items()
            if k not in visual_keys
        }
        normalized_non_visual = (
            original_normalize_dict(non_visual, filtered_stats) if non_visual else {}
        )
 
        result = dict(dict_)
        result.update(normalized_non_visual)
        return result
 
    ObsUtils.normalize_dict = patched_normalize_dict
    try:
        yield
    finally:
        ObsUtils.normalize_dict = original_normalize_dict


@contextmanager
def _checkpoint_saving_patch(*, enabled: bool, train_module):
    if enabled:
        yield
        return

    original_save_model = train_module.TrainUtils.save_model
    original_copyfile = train_module.shutil.copyfile

    def save_model_disabled(*args, **kwargs):
        ckpt_path = kwargs.get("ckpt_path", "<unknown>")
        print(f"[robomimic] checkpoint saving disabled, skipped {ckpt_path}")

    def copyfile_disabled(src, dst, *args, **kwargs):
        if Path(src).name in {"last.pth", "model_latest.pth"}:
            return dst
        return original_copyfile(src, dst, *args, **kwargs)

    train_module.TrainUtils.save_model = save_model_disabled
    train_module.shutil.copyfile = copyfile_disabled
    try:
        yield
    finally:
        train_module.TrainUtils.save_model = original_save_model
        train_module.shutil.copyfile = original_copyfile


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _safe_name(text: str) -> str:
    return text.strip().replace("/", "__").replace(":", "_")


def _repo_short_name(repo_or_path: str) -> str:
    p = Path(repo_or_path).expanduser()
    if p.is_dir():
        return p.resolve().name
    return repo_or_path.split("/", 1)[1] if "/" in repo_or_path else repo_or_path


def _is_local_dir(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return Path(value).expanduser().is_dir()
    except OSError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _obs_profile_from_sensors(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value in {"state", "proprio", "lowdim", "low_dim"}:
            return "lowdim"
        if value in {"camera", "cameras", "rgb", "vision"}:
            return "cameras"
    raise ValueError("sensors must be one of: state, proprio, cameras")


def _low_dim_selection_from_sensors(
    args: argparse.Namespace,
    obs_profile: str | None,
    default_sensors: str | None = None,
) -> str:
    sensors = args.sensors or default_sensors  # --sensors, else the dataset entry's default_sensors
    if sensors in {"proprio", "cameras"}:
        return "proprio"
    if sensors == "state":
        return "state"
    if obs_profile == "cameras":
        return "proprio"
    return "all"


def _camera_names_for_profile(
    args: argparse.Namespace,
    configured_names: tuple[str, ...],
    obs_profile: str | None,
) -> tuple[str, ...] | None:
    if obs_profile != "cameras":
        return None
    return (
        _normalize_camera_names(args.cameras)
        or _normalize_camera_names(configured_names)
        or default_camera_names()
    )


def _normalize_camera_names(values: Any) -> tuple[str, ...]:
    names: list[str] = []
    for value in values or ():
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(names)


def _dataset_obs_profile(
    args: argparse.Namespace,
    raw: dict[str, Any],
    policy_preset: dict[str, Any] | None,
) -> str | None:
    return (
        args.obs_profile
        or _obs_profile_from_sensors(args.sensors)
        or config_store.policy_obs_profile(policy_preset)
        or _obs_profile_from_sensors(raw.get("default_sensors"))
        or raw.get("obs_profile")
    )


def _legacy_architecture_name(config_name: str | None) -> str | None:
    if config_name is None:
        return None
    stem = Path(config_name).stem
    if stem in {"bc_rnn", "bc_rnn_lowdim"}:
        return "rnn"
    if stem in {"bc_transformer", "bc_transformer_lowdim"}:
        return "transformer"
    if stem in {"bc_mlp", "bc_mlp_lowdim", "bc_resnet", "bc_resnet_cameras"}:
        return "mlp"
    return None


def _read_task_from_lerobot(root: Path) -> str:
    env_config = root / "meta" / "env_config.json"
    if not env_config.is_file():
        raise FileNotFoundError(f"Missing LeRobot env config: {env_config}")
    task = _read_json(env_config).get("task")
    if not task:
        raise ValueError(f"{env_config} does not contain a non-empty 'task'")
    return str(task)


def _looks_like_lerobot_root(root: Path) -> bool:
    return (root / "meta" / "env_config.json").is_file() and (
        root / "meta" / "action_layout.json"
    ).is_file()


def _has_downloaded_videos(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return True

    features = _read_json(info_path).get("features", {})
    if not any(item.get("dtype") == "video" for item in features.values()):
        return True

    videos_dir = root / "videos"
    return videos_dir.is_dir() and any(videos_dir.rglob("*.mp4"))


def _find_existing_lerobot_root(datasets_root: Path, safe_repo: str) -> Path | None:
    for candidate in sorted(datasets_root.glob(f"*/lerobot/{safe_repo}")):
        if _looks_like_lerobot_root(candidate):
            return candidate
    return None


def _spec_from_args(
    args: argparse.Namespace,
    policy_preset: dict[str, Any] | None,
) -> DatasetSpec | None:
    dataset_file = _dataset_file_path(args.dataset)
    dataset_selector = args.dataset_name or (
        args.dataset if dataset_file is None else None
    )
    if dataset_selector is None and args.repo_id is None:
        return None

    raw: dict[str, Any] = {}
    repo_id = args.repo_id
    if dataset_selector is not None:
        registry = _dataset_registry()
        if dataset_selector in registry:
            raw = dict(registry[dataset_selector])
        elif repo_id is None and (
            _is_local_dir(dataset_selector) or "/" in dataset_selector
        ):
            repo_id = dataset_selector
        elif repo_id is None:
            known = ", ".join(sorted(registry)) or "<empty registry>"
            raise KeyError(
                f"Unknown dataset '{dataset_selector}'. Known datasets: {known}. "
                "Pass a dataset alias, HDF5 path, Hugging Face repo id, local "
                "LeRobot root, or set --repo-id."
            )
    if not raw:  # a repo id or folder outside the registry trains on state, like every registered set
        raw = {"default_sensors": "state"}

    if dataset_selector is not None and dataset_selector in _dataset_registry():
        name = dataset_selector
    else:  # a repo id ("dexsuite/lift_franka_allegro") or a local folder: its last path part, made file-safe
        name = _safe_name(Path(str(repo_id or dataset_selector).rstrip("/")).name)
    return DatasetSpec(
        name=name,
        repo_id=repo_id or raw.get("repo_id"),
        task=args.task or raw.get("task"),
        obs_yaml=args.obs or raw.get("obs_yaml"),
        obs_profile=_dataset_obs_profile(args, raw, policy_preset),
        camera_names=tuple(str(name) for name in raw.get("camera_names") or ()),
        obs_exclude_keys=tuple(str(key) for key in raw.get("obs_exclude_keys", ())),
        h5_basename=args.h5_basename or raw.get("h5_basename"),
        action_normalization=args.action_normalization
        or str(raw.get("action_normalization", "dataset")),
        revision=args.revision or raw.get("revision"),
        no_images=_resolve_no_images(args, raw, policy_preset),
        default_sensors=raw.get("default_sensors"),
    )


def _resolve_no_images(
    args: argparse.Namespace,
    dataset_spec: dict[str, Any],
    policy_preset: dict[str, Any] | None,
) -> bool | None:
    if args.no_images:
        return True
    if args.sensors == "cameras":
        return False
    if args.sensors in {"state", "proprio"}:
        return True

    dataset_sensors = dataset_spec.get("default_sensors")
    if dataset_sensors == "cameras":
        return False
    if dataset_sensors in {"state", "proprio"}:
        return True

    if "no_images" in dataset_spec and dataset_spec["no_images"] is not None:
        return bool(dataset_spec["no_images"])
    policy_no_images = config_store.policy_no_images(policy_preset)
    if policy_no_images is not None:
        return policy_no_images
    return None


def _download_or_reuse_lerobot(
    spec: DatasetSpec,
    datasets_root: Path,
    *,
    include_videos: bool,
    force_download: bool,
    download_workers: int,
) -> tuple[Path, str]:
    if spec.repo_id is None:
        raise ValueError(
            "Dataset preparation requires a repo id or local LeRobot path. "
            "Set DATASET_REGISTRY[dataset_name]['repo_id'] or pass --repo-id."
        )

    if _is_local_dir(spec.repo_id):
        local_root = _expand(spec.repo_id)
        if not _looks_like_lerobot_root(local_root):
            raise FileNotFoundError(
                f"Local dataset does not look like a LeRobot root: {local_root}"
            )
        task = spec.task or _read_task_from_lerobot(local_root)
        return local_root, task

    safe_repo = _safe_name(spec.repo_id)
    existing_root = (
        _find_existing_lerobot_root(datasets_root, safe_repo)
        if spec.task is None
        else None
    )
    if existing_root is not None:
        if include_videos and not _has_downloaded_videos(existing_root):
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=str(existing_root),
                force_download=force_download,
                max_workers=download_workers,
            )
        task = _read_task_from_lerobot(existing_root)
        return existing_root, task

    if spec.task is not None:
        raw_root = datasets_root / spec.task / "lerobot" / safe_repo
    else:
        raw_root = datasets_root / "_downloads" / safe_repo

    if raw_root.exists() and force_download:
        shutil.rmtree(raw_root)

    needs_download = not _looks_like_lerobot_root(raw_root) or (
        include_videos and not _has_downloaded_videos(raw_root)
    )
    if needs_download:
        raw_root.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=str(raw_root),
            force_download=force_download,
            max_workers=download_workers,
            ignore_patterns=None if include_videos else ["videos/**"],
        )

    task = spec.task or _read_task_from_lerobot(raw_root)
    final_root = datasets_root / task / "lerobot" / safe_repo
    if raw_root.resolve() != final_root.resolve():
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            if force_download:
                shutil.rmtree(final_root)
            else:
                return final_root, task
        shutil.move(str(raw_root), str(final_root))
        downloads = datasets_root / "_downloads"
        if downloads.exists() and not any(downloads.iterdir()):
            downloads.rmdir()
    return final_root, task


def prepare_dataset_from_registry(
    args: argparse.Namespace,
    policy_preset: dict[str, Any] | None,
) -> PreparedDataset | None:
    spec = _spec_from_args(args, policy_preset)
    if spec is None:
        return None

    datasets_root = _expand(args.datasets_root)
    no_images = bool(spec.no_images) or bool(args.no_images)
    raw_root, task = _download_or_reuse_lerobot(
        spec,
        datasets_root,
        include_videos=not no_images,
        force_download=args.force_download,
        download_workers=args.download_workers,
    )

    task_root = datasets_root / task
    hdf5_dir = task_root / "hdf5"
    configs_dir = task_root / "configs"
    runs_dir = _expand(args.runs_root) / task

    h5_basename = spec.h5_basename or _repo_short_name(spec.repo_id or spec.name)
    hdf5_path = hdf5_dir / f"{h5_basename}.hdf5"
    if args.force_convert or not hdf5_path.is_file():
        needs_visual_reconvert = False
    else:
        try:
            needs_visual_reconvert = spec.obs_profile == "cameras" and not hdf5_has_visual_observations(hdf5_path)
        except OSError:  # unreadable (an interrupted older conversion): convert again
            needs_visual_reconvert = True
    if args.force_convert or not hdf5_path.is_file() or needs_visual_reconvert:
        compression = None if args.compression == "none" else args.compression
        _converter_module().convert_lerobot_to_hdf5(
            repo_or_root=str(raw_root),
            output_path=str(hdf5_dir),
            split=args.split,
            h5_basename=h5_basename,
            video_backend=args.video_backend,
            allow_fallback=args.allow_fallback,
            read_workers=args.read_workers or (1 if spec.obs_profile == "cameras" else 8),
            block_size=args.block_size,
            compression=compression,
            compression_opts=args.compression_opts if compression else None,
            no_images=no_images,
        )

    action_config_path = (
        Path(args.action_config).expanduser().resolve()
        if args.action_config
        else hdf5_dir / "action_config.json"
    )
    if not action_config_path.is_file():
        raise FileNotFoundError(f"Missing action config: {action_config_path}")

    obs_yaml_path = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=spec.obs_yaml,
        obs_profile=spec.obs_profile,
        exclude_keys=spec.obs_exclude_keys,
        camera_names=_camera_names_for_profile(
            args, spec.camera_names, spec.obs_profile
        ),
        low_dim_selection=_low_dim_selection_from_sensors(args, spec.obs_profile, spec.default_sensors),
        task=task,
    )
    return PreparedDataset(
        key=spec.name,
        task=task,
        task_root=task_root,
        raw_root=raw_root,
        hdf5_dir=hdf5_dir,
        hdf5_path=hdf5_path,
        action_config_path=action_config_path,
        action_normalization=spec.action_normalization,
        obs_yaml_path=obs_yaml_path,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
    )


def prepared_from_hdf5(
    args: argparse.Namespace,
    policy_preset: dict[str, Any] | None,
) -> PreparedDataset | None:
    hdf5_path = _dataset_file_path(args.dataset)
    if hdf5_path is None:
        return None

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found: {hdf5_path}")

    hdf5_dir = hdf5_path.parent
    task_root = hdf5_dir.parent if hdf5_dir.name == "hdf5" else hdf5_dir
    task = args.task or task_root.name
    action_config_path = (
        Path(args.action_config).expanduser().resolve()
        if args.action_config
        else hdf5_dir / "action_config.json"
    )
    obs_profile = (
        args.obs_profile
        or _obs_profile_from_sensors(args.sensors)
        or config_store.policy_obs_profile(policy_preset)
    )
    obs_yaml_path = resolve_observation_yaml(
        hdf5_path=hdf5_path,
        hdf5_dir=hdf5_dir,
        obs_yaml=args.obs,
        obs_profile=obs_profile,
        camera_names=_camera_names_for_profile(args, (), obs_profile),
        low_dim_selection=_low_dim_selection_from_sensors(args, obs_profile),
        task=task,
    )
    return PreparedDataset(
        key=args.dataset_name or hdf5_path.stem,
        task=task,
        task_root=task_root,
        raw_root=None,
        hdf5_dir=hdf5_dir,
        hdf5_path=hdf5_path,
        action_config_path=action_config_path,
        action_normalization=args.action_normalization or "dataset",
        obs_yaml_path=obs_yaml_path,
        configs_dir=task_root / "configs",
        runs_dir=_expand(args.runs_root) / task,
    )


def _load_base_config(
    args: argparse.Namespace,
    policy_preset: dict[str, Any] | None,
):
    register_custom_algorithms()
    preset_algo_name = config_store.policy_algo_name(policy_preset)
    config_file = _config_file_path(args.config)
    if config_file is not None:
        ext_cfg = _read_json(config_file)
        if preset_algo_name is not None and preset_algo_name != ext_cfg["algo_name"]:
            raise ValueError(
                f"Policy preset uses algo '{preset_algo_name}', but config uses "
                f"algo '{ext_cfg['algo_name']}'."
            )
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        return config

    if (
        args.algo is not None
        and preset_algo_name is not None
        and args.algo != preset_algo_name
    ):
        raise ValueError(
            f"--algo {args.algo} conflicts with policy preset algo {preset_algo_name}."
        )

    algo_name = args.algo or preset_algo_name
    if algo_name is None:
        raise ValueError("--algo or --config preset is required")
    return config_factory(algo_name)


def _apply_presets_to_config(
    config,
    policy_preset: dict[str, Any] | None,
    rollout_preset: dict[str, Any] | None,
) -> None:
    with config.values_unlocked():
        config_store.apply_preset_config(config, policy_preset)
        config_store.apply_preset_config(config, rollout_preset)


def _apply_bc_architecture(config, args: argparse.Namespace) -> None:
    if config.algo_name != "bc":
        return

    architecture = (
        args.architecture
        or _legacy_architecture_name(args.config)
        or _legacy_architecture_name(args.policy_preset)
    )
    if architecture is None:
        if Path(str(args.config or "bc")).stem != "bc":
            return  # a tuned preset (gmm_best, gaussian_best, ...) already says what it trains: keep it
        architecture = "mlp"
    with config.values_unlocked():
        config.algo.rnn.enabled = architecture == "rnn"
        config.algo.transformer.enabled = architecture == "transformer"
        config.algo.gmm.enabled = architecture == "transformer"

        if architecture == "mlp":
            config.train.seq_length = 1
            config.train.frame_stack = 1
            config.algo.actor_layer_dims = [1024, 1024]
        elif architecture == "rnn":
            config.train.seq_length = 10
            config.train.frame_stack = 1
            config.algo.actor_layer_dims = [1024, 1024]
            config.algo.rnn.horizon = 10
        elif architecture == "transformer":
            config.train.seq_length = 1
            config.train.frame_stack = 10
            config.algo.actor_layer_dims = []
            config.algo.transformer.context_length = 10


def _apply_act_overrides(config, args: argparse.Namespace) -> None:
    if config.algo_name != "act":
        return

    observation_horizon = args.obs_horizon or config.algo.horizon.observation_horizon
    action_horizon = args.action_horizon or config.algo.horizon.action_horizon

    with config.values_unlocked():
        config.algo.horizon.observation_horizon = int(observation_horizon)
        config.algo.horizon.action_horizon = int(action_horizon)
        config.train.frame_stack = int(observation_horizon)
        config.train.seq_length = int(action_horizon)
        config.train.hdf5_load_next_obs = False
        config.train.dataset_keys = ["actions"]

        if args.act_embed_dim is not None:
            config.algo.transformer.embed_dim = args.act_embed_dim
        if args.act_num_layers is not None:
            config.algo.transformer.num_layers = args.act_num_layers
        if args.act_num_heads is not None:
            config.algo.transformer.num_heads = args.act_num_heads
        if args.act_dropout is not None:
            config.algo.transformer.emb_dropout = args.act_dropout
            config.algo.transformer.attn_dropout = args.act_dropout
            config.algo.transformer.block_output_dropout = args.act_dropout
        if args.act_lr is not None:
            config.algo.optim_params.policy.learning_rate.initial = args.act_lr

    _validate_act_config(config)

def _apply_diffusion_overrides(config, args: argparse.Namespace) -> None:
    if config.algo_name != "diffusion_policy":
        return
 
    with config.values_unlocked():
        if getattr(args, "obs_horizon", None) is not None:
            config.algo.horizon.observation_horizon = int(args.obs_horizon)
            config.train.frame_stack = int(args.obs_horizon)
        if getattr(args, "action_horizon", None) is not None:
            config.algo.horizon.action_horizon = int(args.action_horizon)
        if getattr(args, "diffusion_prediction_horizon", None) is not None:
            config.algo.horizon.prediction_horizon = int(args.diffusion_prediction_horizon)
            config.train.seq_length = int(args.diffusion_prediction_horizon)


def _validate_act_config(config) -> None:
    observation_horizon = int(config.algo.horizon.observation_horizon)
    action_horizon = int(config.algo.horizon.action_horizon)
    if int(config.train.frame_stack) != observation_horizon:
        raise ValueError(
            "ACT requires train.frame_stack to match algo.horizon.observation_horizon."
        )
    if int(config.train.seq_length) != action_horizon:
        raise ValueError(
            "ACT requires train.seq_length to match algo.horizon.action_horizon."
        )
    if config.train.hdf5_load_next_obs:
        raise ValueError(
            "ACT does not use next observations. Set hdf5_load_next_obs=false."
        )
    if "actions" not in config.train.dataset_keys:
        raise ValueError("ACT requires actions in train.dataset_keys.")
    embed_dim = int(config.algo.transformer.embed_dim)
    num_heads = int(config.algo.transformer.num_heads)
    if embed_dim % num_heads != 0:
        raise ValueError("ACT transformer embed_dim must be divisible by num_heads.")


def _set_config_child(parent, key: str, value: Any) -> None:
    if isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


def _apply_sensor_encoder_overrides(config, args: argparse.Namespace) -> None:
    if args.sensors != "cameras":
        return
 
    backbone = args.backbone or "ResNet18Conv"
    crop_size = int(args.crop_size)
 
    with config.values_unlocked():
        encoder = config.observation.encoder.rgb
        encoder.core_class = "VisualCore"
        _set_config_child(encoder.core_kwargs, "feature_dimension", 64)
        _set_config_child(encoder.core_kwargs, "backbone_class", backbone)
        _set_config_child(
            encoder.core_kwargs,
            "backbone_kwargs",
            {"pretrained": bool(args.pretrained), "input_coord_conv": False},
        )
        _set_config_child(encoder.core_kwargs, "pool_class", "SpatialSoftmax")
        _set_config_child(
            encoder.core_kwargs,
            "pool_kwargs",
            {
                "num_kp": 32,
                "learnable_temperature": False,
                "temperature": 1.0,
                "noise_std": 0.0,
            },
        )
        encoder.obs_randomizer_class = "CropRandomizer"
        _set_config_child(encoder.obs_randomizer_kwargs, "crop_height", crop_size)
        _set_config_child(encoder.obs_randomizer_kwargs, "crop_width", crop_size)
        _set_config_child(encoder.obs_randomizer_kwargs, "num_crops", 1)
        _set_config_child(encoder.obs_randomizer_kwargs, "pos_enc", False)


def _apply_prepared_dataset_to_config(
    config,
    prepared: PreparedDataset | None,
    args: argparse.Namespace,
) -> Path | None:
    apply_dataset_config = prepared is not None and (
        args.config is None or _config_is_preset(args) or args.apply_dataset_config
    )

    generated_config_path = None
    with config.unlocked():
        if prepared is not None:
            config.train.data = [{"path": str(prepared.hdf5_path)}]
            config.train.output_dir = str(prepared.runs_dir)
        else:
            hdf5_path = _dataset_file_path(args.dataset)
            if hdf5_path is not None:
                config.train.data = [{"path": str(hdf5_path)}]

        if args.name is not None:
            config.experiment.name = args.name
        elif prepared is not None and (args.config is None or _config_is_preset(args)):
            config.experiment.name = (
                f"{prepared.key}_{args.policy_preset or config.algo_name}"
            )

        if args.output_dir is not None:
            config.train.output_dir = str(_expand(args.output_dir))

        if apply_dataset_config:
            if not prepared.action_config_path.is_file():
                raise FileNotFoundError(
                    f"Missing action config: {prepared.action_config_path}"
                )
            if prepared.obs_yaml_path is None:
                raise ValueError(
                    "No obs YAML was resolved. Pass --obs, --obs-profile, or add "
                    "'obs_yaml' / 'obs_profile' to config/datasets.yaml."
                )

            action_keys, action_config = (
                _config_module().build_action_settings_from_json(
                    str(prepared.action_config_path),
                    action_normalization=prepared.action_normalization,
                )
            )
            config.train.action_keys = action_keys
            config.train.action_config = action_config
            _config_module().apply_observation_yaml(config, str(prepared.obs_yaml_path))
            config.train.hdf5_cache_mode = (
                None if any("/" in key for key in action_keys) else "low_dim"
            )
            config.train.hdf5_use_swmr = True

    if prepared is not None:
        prepared.configs_dir.mkdir(parents=True, exist_ok=True)
        filename = args.config_name or f"{config.experiment.name}.json"
        generated_config_path = prepared.configs_dir / filename
        with generated_config_path.open("w") as f:
            json.dump(config, f, indent=4)

        if prepared.obs_yaml_path is not None:
            _config_module().enforce_obs_keys(
                str(prepared.hdf5_path),
                str(prepared.obs_yaml_path),
                strict=args.strict_obs_keys,
            )
        if apply_dataset_config:
            _config_module().enforce_action_keys(
                str(prepared.hdf5_path),
                list(config.train.action_keys),
                strict=args.strict_action_keys,
            )

    return generated_config_path


def _apply_debug_overrides(config):
    with config.values_unlocked():
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10


def _apply_run_overrides(config, args: argparse.Namespace) -> None:
    with config.values_unlocked():
        if getattr(args, "num_epochs", None) is not None:
            config.train.num_epochs = int(args.num_epochs)
        if getattr(args, "batch_size", None) is not None:
            config.train.batch_size = int(args.batch_size)
        if getattr(args, "epoch_every_n_steps", None) is not None:
            config.experiment.epoch_every_n_steps = int(args.epoch_every_n_steps)
        if getattr(args, "validation_epoch_every_n_steps", None) is not None:
            config.experiment.validation_epoch_every_n_steps = int(
                args.validation_epoch_every_n_steps,
            )
        if getattr(args, "hdf5_cache_mode", None) is not None:
            config.train.hdf5_cache_mode = (
                None if args.hdf5_cache_mode == "none" else args.hdf5_cache_mode
            )
        if getattr(args, "hdf5_normalize_obs", None) is not None:
            config.train.hdf5_normalize_obs = bool(args.hdf5_normalize_obs)
        if getattr(args, "train_dataset_keys", None) is not None:
            config.train.dataset_keys = list(args.train_dataset_keys)
        if getattr(args, "lr", None) is not None:
            config.algo.optim_params.policy.learning_rate.initial = float(args.lr)
        if getattr(args, "bc_gaussian", None) and getattr(args, "bc_gmm", None):
            raise ValueError("--bc-gaussian and --bc-gmm are mutually exclusive.")
        if getattr(args, "bc_gaussian", None) is not None and config.algo_name == "bc":
            config.algo.gaussian.enabled = bool(args.bc_gaussian)
            if args.bc_gaussian:
                config.algo.gmm.enabled = False
                config.algo.vae.enabled = False
        if getattr(args, "bc_gmm", None) is not None and config.algo_name == "bc":
            config.algo.gmm.enabled = bool(args.bc_gmm)
            if args.bc_gmm:
                config.algo.gaussian.enabled = False
                config.algo.vae.enabled = False
        if getattr(args, "save_models", None) is not None:
            config.experiment.save.enabled = bool(args.save_models)
        if getattr(args, "render_video", None) is not None:
            config.experiment.render_video = bool(args.render_video)
        if getattr(args, "video_skip", None) is not None:
            config.experiment.video_skip = int(args.video_skip)
        if getattr(args, "rollout_enabled", None) is not None:
            config.experiment.rollout.enabled = bool(args.rollout_enabled)
        if getattr(args, "rollout_n", None) is not None:
            config.experiment.rollout.n = int(args.rollout_n)
        if getattr(args, "rollout_horizon", None) is not None:
            config.experiment.rollout.horizon = int(args.rollout_horizon)
        if getattr(args, "rollout_rate", None) is not None:
            config.experiment.rollout.rate = int(args.rollout_rate)
        if getattr(args, "rollout_warmstart", None) is not None:
            config.experiment.rollout.warmstart = int(args.rollout_warmstart)
        if getattr(args, "rollout_terminate_on_success", None) is not None:
            config.experiment.rollout.terminate_on_success = bool(
                args.rollout_terminate_on_success,
            )
        if getattr(args, "seed", None) is not None:
            config.train.seed = int(args.seed)
        if getattr(args, "wandb", False):
            config.experiment.logging.log_wandb = True
        if getattr(args, "wandb_project", None) is not None:
            config.experiment.logging.wandb_proj_name = str(args.wandb_project)


def build_config(
    args: argparse.Namespace,
    prepared: PreparedDataset | None,
    policy_preset: dict[str, Any] | None,
    rollout_preset: dict[str, Any] | None,
):
    config = _load_base_config(args, policy_preset)
    _apply_presets_to_config(config, policy_preset, rollout_preset)
    _apply_bc_architecture(config, args)
    _apply_act_overrides(config, args)
    _apply_diffusion_overrides(config, args)
    _apply_sensor_encoder_overrides(config, args)
    generated_config_path = _apply_prepared_dataset_to_config(config, prepared, args)
    _apply_run_overrides(config, args)

    if args.debug:
        _apply_debug_overrides(config)

    if generated_config_path is not None:
        with generated_config_path.open("w") as f:
            json.dump(config, f, indent=4)

    config.lock()
    return config, generated_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DexSuite LeRobot dataset and train a robomimic policy.",
    )

    # Upstream robomimic train arguments.
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Policy recipe name, such as bc or diffusion, or raw robomimic JSON.",
    )
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument(
        "--policy-preset", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--rollout-preset", "--rollout", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset alias, Hugging Face repo id, local LeRobot root, or HDF5 path.",
    )
    parser.add_argument(
        "--architecture",
        choices=["mlp", "rnn", "transformer"],
        default=None,
        help="BC policy architecture. Defaults to mlp for --config bc.",
    )
    parser.add_argument(
        "--sensors",
        choices=["state", "proprio", "cameras"],
        default=None,
        help=(
            "Observation source. State uses the task low-dimensional profile, "
            "proprio uses robot keys, cameras adds RGB plus robot proprio."
        ),
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Camera names to include when --sensors cameras is used.",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Visual backbone class for camera policies, for example ResNet18Conv.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=196,
        help=(
            "CropRandomizer crop height/width in pixels, for --sensors "
            "cameras. Default 196 = the paper's image runs on DexSuite's "
            "224x224 renders (~87%%). robomimic's own 76px default was tuned "
            "for 84x84 images and would discard ~2/3 of a DexSuite frame."
        ),
    )

    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pretrained visual backbone weights when supported.",
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=None,
        help="ACT observation horizon. Also sets train.frame_stack.",
    )
    parser.add_argument(
        "--action-horizon",
        "--chunk-size",
        type=int,
        default=None,
        help="ACT action chunk horizon. Also sets train.seq_length.",
    )
    parser.add_argument(
        "--diffusion-prediction-horizon",
        type=int,
        default=None,
        help="Diffusion policy prediction_horizon override.",
    )
    parser.add_argument(
        "--act-embed-dim",
        type=int,
        default=None,
        help="ACT transformer embedding dimension.",
    )
    parser.add_argument(
        "--act-num-layers",
        type=int,
        default=None,
        help="ACT transformer layer count.",
    )
    parser.add_argument(
        "--act-num-heads",
        type=int,
        default=None,
        help="ACT transformer attention head count.",
    )
    parser.add_argument(
        "--act-dropout",
        type=float,
        default=None,
        help="ACT transformer dropout for embedding, attention, and block output.",
    )
    parser.add_argument(
        "--act-lr",
        type=float,
        default=None,
        help="ACT policy optimizer learning rate.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for robomimic config.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epoch-every-n-steps", type=int, default=None)
    parser.add_argument("--validation-epoch-every-n-steps", type=int, default=None)
    parser.add_argument(
        "--hdf5-cache-mode",
        choices=["all", "low_dim", "none"],
        default=None,
    )
    parser.add_argument(
        "--hdf5-normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--train-dataset-keys", nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--bc-gaussian",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--bc-gmm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use a GMM policy head for BC instead of a unimodal Gaussian. "
            "Works with any --architecture (mlp, rnn, transformer). "
            "Mutually exclusive with --bc-gaussian."
        ),
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--video-skip", type=int, default=None)
    parser.add_argument(
        "--rollout-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--rollout-n", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--rollout-rate", type=int, default=None)
    parser.add_argument("--rollout-warmstart", type=int, default=None)
    parser.add_argument(
        "--rollout-terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Dataset registry / loading arguments.
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--datasets-root", type=str, default="datasets")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/imitation_learning/robomimic",
        help="Root directory for robomimic training outputs.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--force-download", action="store_true")

    # Conversion arguments.
    parser.add_argument("--split", default="train")
    parser.add_argument("--h5-basename", type=str, default=None)
    parser.add_argument(
        "--video-backend",
        default="auto",
        choices=["auto", "pyav", "video_reader", "decord"],
    )
    parser.add_argument("--no-fallback", dest="allow_fallback", action="store_false")
    parser.add_argument("--no-images", "--no_image", action="store_true")
    parser.add_argument(
        "--read-workers",
        type=int,
        default=None,
        help="Parallel readers for the LeRobot to HDF5 conversion. Default 1 when camera videos are decoded "
        "(the video decoder is not thread safe), 8 otherwise.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--compression", default="none", choices=["none", "gzip"])
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument("--force-convert", action="store_true")

    # Config generation arguments.
    parser.add_argument("--obs", type=str, default=None)
    parser.add_argument("--obs-profile", choices=["lowdim", "cameras"], default=None)
    parser.add_argument("--action-config", type=str, default=None)
    parser.add_argument(
        "--action-normalization",
        choices=["dataset", "none", "min_max", "gaussian"],
        default=None,
        help="Override dataset action normalization when building robomimic config.",
    )
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--apply-dataset-config",
        action="store_true",
        help="Also merge action_config.json and obs YAML when --config is supplied.",
    )
    parser.add_argument("--strict-obs-keys", action="store_true")
    parser.add_argument("--strict-action-keys", action="store_true")
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print dataset aliases from config/datasets.yaml and exit.",
    )
    parser.add_argument(
        "--list-policy-presets",
        action="store_true",
        help="Print policy preset names and exit.",
    )
    parser.add_argument(
        "--list-rollout-presets",
        action="store_true",
        help="Print rollout preset names and exit.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset and generated config, then exit before training.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.list_datasets:
        for key, spec in sorted(_dataset_registry().items()):
            repo = spec.get("repo_id", "<no repo>")
            task = spec.get("task", "<no task>")
            print(f"{key}: {repo} ({task})")
        return
    if args.list_policy_presets:
        for key in config_store.list_policy_presets():
            print(
                config_store.preset_summary(key, config_store.load_policy_preset(key))
            )
        return
    if args.list_rollout_presets:
        for key in config_store.list_rollout_presets():
            print(
                config_store.preset_summary(key, config_store.load_rollout_preset(key))
            )
        return

    policy_preset = _load_policy_preset(args)
    rollout_preset = _load_rollout_preset(args)

    prepared = prepare_dataset_from_registry(args, policy_preset)
    if (
        prepared is None
        and (args.dataset is not None)
        and (
            args.config is None or _config_is_preset(args) or args.apply_dataset_config
        )
    ):
        prepared = prepared_from_hdf5(args, policy_preset)

    config, generated_config_path = build_config(
        args,
        prepared,
        policy_preset,
        rollout_preset,
    )

    if prepared is not None:
        if args.policy_preset is not None:
            print(f"[preset] policy: {args.policy_preset}")
        if args.rollout_preset is not None:
            print(f"[preset] rollout: {args.rollout_preset}")
        print(f"[dataset] task root: {prepared.task_root}")
        if prepared.raw_root is not None:
            print(f"[dataset] lerobot: {prepared.raw_root}")
        print(f"[dataset] hdf5: {prepared.hdf5_path}")
        print(f"[dataset] action config: {prepared.action_config_path}")
        print(f"[dataset] action normalization: {prepared.action_normalization}")
        if prepared.obs_yaml_path is not None:
            print(f"[dataset] obs yaml: {prepared.obs_yaml_path}")
        if generated_config_path is not None:
            print(f"[config] effective config: {generated_config_path}")
        print(f"[models] output dir: {config.train.output_dir}")

    if args.prepare_only:
        return

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    result = "finished run successfully!"
    try:
        _robomimic_train(config, device=device, resume=args.resume)
    except Exception as exc:
        result = f"run failed with error:\n{exc}\n\n{traceback.format_exc()}"
        print(result)
        sys.exit(1)
    print(result)


if __name__ == "__main__":
    main(parse_args())

