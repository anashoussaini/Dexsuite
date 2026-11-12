"""Convert LeRobot datasets to robomimic HDF5 datasets.

Converted files store flat robomimic actions, split action component datasets,
rewards, dones, empty states, observations, and the dataset metadata robomimic
uses to reconstruct DexSuite environments. Action components are derived from
`meta/action_layout.json` and controller details from `meta/env_config.json`.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from huggingface_hub import HfFileSystem
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_NON_TC_BACKENDS = ("pyav", "video_reader", "decord")


# ------------------------- IO helpers -------------------------


def is_hf_repo_id(s: str) -> bool:
    try:
        return ("/" in s) and (not Path(s).expanduser().is_dir())
    except Exception:
        return True


def load_env_config(repo_or_root: str) -> dict[str, Any]:
    if is_hf_repo_id(repo_or_root):
        fs = HfFileSystem()
        with fs.open(f"datasets/{repo_or_root}/meta/env_config.json", "r") as f:
            return json.load(f)
    else:
        with (Path(repo_or_root) / "meta" / "env_config.json").open() as f:
            return json.load(f)


def load_action_layout(repo_or_root: str) -> dict[str, Any]:
    if is_hf_repo_id(repo_or_root):
        fs = HfFileSystem()
        with fs.open(f"datasets/{repo_or_root}/meta/action_layout.json", "r") as f:
            return json.load(f)
    else:
        p = Path(repo_or_root) / "meta" / "action_layout.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        with p.open() as f:
            return json.load(f)


def to_np1(x, dtype) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        arr = arr.reshape(1)
    return arr.astype(dtype, copy=False)


def obs_storage_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    return arr


# ------------------------- obs/image helpers -------------------------


def is_image_like(v: Any) -> bool:
    try:
        a = np.asarray(v)
    except Exception:
        return False
    if a.ndim == 3:
        if a.shape[-1] in (1, 3, 4):
            return True
        if a.shape[0] in (1, 3, 4):
            return True
    return a.ndim == 4 and (a.shape[-1] in (1, 3, 4) or a.shape[1] in (1, 3, 4))


def is_numeric_array(v: Any) -> bool:
    try:
        a = np.asarray(v)
    except Exception:
        return False
    if a.dtype == np.uint8:
        return True
    return np.issubdtype(a.dtype, np.number) or (a.dtype == np.bool_)


def ensure_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    # CHW -> add time dim
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
        a = a[None, ...]  # (1,H,W,C)
    if a.ndim == 4:
        if a.shape[1] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
            a = np.transpose(a, (0, 2, 3, 1))
        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating):
                mx = float(a.max()) if a.size else 1.0
                if mx <= 1.01:
                    a = (a * 255.0).round()
                a = np.clip(a, 0, 255).astype(np.uint8)
            else:
                a = a.astype(np.uint8)
        if a.ndim != 4 or a.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected (T,H,W,C) uint8-ish, got {a.shape} {a.dtype}")
    return a


def obs_key_clean(k: str) -> str:
    if k.startswith("obs."):
        k = k.split("obs.", 1)[1]
    return k.replace(".", "_")


def repo_id_from_local_root(root: Path) -> str:
    """Recover a LeRobot repo id from a local snapshot folder name."""
    if "__" in root.name:
        owner, repo = root.name.split("__", 1)
        return f"{owner}/{repo}"
    return root.name


def _local_lerobot_root(repo_or_root: str) -> Path | None:
    if is_hf_repo_id(repo_or_root):
        return None
    root = Path(repo_or_root).expanduser().resolve()
    if not (root / "data").is_dir() or not (root / "meta").is_dir():
        return None
    return root


def _local_parquet_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet"))


def _load_local_features(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    features = info.get("features", {})
    return features if isinstance(features, dict) else {}


def _is_visual_column(key: str, features: dict[str, Any]) -> bool:
    lower_key = key.lower()
    feature = features.get(key, {})
    dtype = feature.get("dtype") if isinstance(feature, dict) else None
    return dtype == "video" or lower_key.startswith("obs.cameras.")


def _local_parquet_columns(root: Path, *, no_images: bool) -> list[str]:
    import pyarrow.parquet as pq

    parquet_files = _local_parquet_files(root)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    schema_names = pq.ParquetFile(parquet_files[0]).schema_arrow.names
    features = _load_local_features(root)
    columns: list[str] = []
    for key in schema_names:
        if key not in {
            "action",
            "reward",
            "success",
            "truncated",
            "failure",
            "episode_index",
        } and not key.startswith("obs."):
            continue
        if no_images and _is_visual_column(key, features):
            continue
        columns.append(key)

    if "episode_index" not in columns:
        raise KeyError("Local parquet dataset is missing required 'episode_index'.")
    if "action" not in columns:
        raise KeyError("Local parquet dataset is missing required 'action'.")
    return columns


def _iter_local_parquet_rows(
    root: Path,
    *,
    columns: list[str],
    batch_size: int,
):
    import pyarrow.parquet as pq

    row_index = 0
    for parquet_file in _local_parquet_files(root):
        parquet = pq.ParquetFile(parquet_file)
        present = [key for key in columns if key in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(
            batch_size=max(1, int(batch_size)),
            columns=present,
        ):
            for row in batch.to_pylist():
                yield row_index, row
                row_index += 1


# ------------------------- action components (flat names) -------------------------


@dataclass
class Component:
    key: str  # "action/<flat_component_name>" e.g. "action/eef_pos", "action/left_gripper"
    start: int
    stop: int
    width: int
    controller: str
    side: str | None  # "left" | "right" | None
    kind: str  # "pos"|"rot"|"quat"|"gripper"|"generic"
    normalized_in_env: bool
    recommended_norm: str | None  # "none"|"min_max"|"gaussian"|None
    rot_format: str | None  # "axis_angle"|"quat_wxyz"|None


def _robot_node(env: dict[str, Any], side: str | None) -> dict[str, Any]:
    robot = env.get("options", {}).get("robot", {})
    if side in ("left", "right"):
        return robot.get(side, {}) or {}
    return robot.get("single", {}) or {}


def _ctrl_info(env: dict[str, Any], role: str, side: str | None) -> tuple[str, bool]:
    node = _robot_node(env, side)
    key = "manipulator_controller" if role == "manipulator" else "gripper_controller"
    ctrl = node.get(key, {}) or {}
    name = str(
        ctrl.get("name")
        or ctrl.get("type")
        or ctrl.get("controller")
        or "UnknownController",
    )
    normalized = bool(ctrl.get("config", {}).get("normalized", False))
    return name, normalized


def _side_role(seg_name: str) -> tuple[str | None, str]:
    toks = str(seg_name).split(".")
    if len(toks) >= 2 and toks[0] in ("left", "right"):
        return toks[0], toks[-1]
    return None, toks[-1]


def _rec_norm(ctrl_lower: str, kind: str, normalized_flag: bool) -> str | None:
    if normalized_flag:
        return "none"
    if "osc" in ctrl_lower or "ik" in ctrl_lower:
        if kind == "pos":
            return "min_max"
        if kind == "rot":
            return "gaussian"
        if kind == "quat":
            return "none"  # do not normalize quaternions
    if "joint_position" in ctrl_lower or "grip" in ctrl_lower:
        return "min_max"
    if kind == "generic":
        return "gaussian"
    return None


def _flat_name(side: str | None, leaf: str) -> str:
    return f"{side}_{leaf}" if side in ("left", "right") else leaf


def _path_for(side: str | None, leaf: str) -> str:
    return f"action/{_flat_name(side, leaf)}"


def build_components(layout: dict[str, Any], env: dict[str, Any]) -> list[Component]:
    segs = sorted(layout.get("segments", []), key=lambda s: int(s.get("start", 0)))
    comps: list[Component] = []
    for seg in segs:
        name_raw = str(seg.get("name", "segment"))
        side, role = _side_role(name_raw)
        start = int(seg.get("start", 0))
        stop = int(seg.get("stop", start))
        ctrl_typ = str(seg.get("ctrl_type", ""))

        if role in ("manipulator", "gripper"):
            ctrl_name, norm_flag = _ctrl_info(env, role, side)
        else:
            ctrl_name, norm_flag = (ctrl_typ or "UnknownController", False)

        ctrl_l = ctrl_name.lower()

        if role == "manipulator":
            # absolute quaternion controller?
            is_abs_quat = (
                "osc" in ctrl_l
                and "quat" in ctrl_l
                and ("abs" in ctrl_l or "absolute" in ctrl_l)
            )
            if is_abs_quat:
                end_pos = min(start + 3, stop)
                end_quat = min(end_pos + 4, stop)
                # pos
                comps.append(
                    Component(
                        key=_path_for(side, "eef_pos"),
                        start=start,
                        stop=end_pos,
                        width=end_pos - start,
                        controller=ctrl_name,
                        side=side,
                        kind="pos",
                        normalized_in_env=norm_flag,
                        recommended_norm=_rec_norm(ctrl_l, "pos", norm_flag),
                        rot_format=None,
                    ),
                )
                # quat
                if end_quat > end_pos:
                    comps.append(
                        Component(
                            key=_path_for(side, "eef_quat"),
                            start=end_pos,
                            stop=end_quat,
                            width=end_quat - end_pos,
                            controller=ctrl_name,
                            side=side,
                            kind="quat",
                            normalized_in_env=norm_flag,
                            recommended_norm=_rec_norm(ctrl_l, "quat", norm_flag),
                            rot_format="quat_wxyz",
                        ),
                    )
                # extras if any
                if stop > end_quat:
                    comps.append(
                        Component(
                            key=_path_for(side, "extra"),
                            start=end_quat,
                            stop=stop,
                            width=stop - end_quat,
                            controller=ctrl_name,
                            side=side,
                            kind="generic",
                            normalized_in_env=norm_flag,
                            recommended_norm=_rec_norm(ctrl_l, "generic", norm_flag),
                            rot_format=None,
                        ),
                    )
            else:
                end_pos = min(start + 3, stop)
                end_rot = min(end_pos + 3, stop)
                comps.append(
                    Component(
                        key=_path_for(side, "eef_pos"),
                        start=start,
                        stop=end_pos,
                        width=end_pos - start,
                        controller=ctrl_name,
                        side=side,
                        kind="pos",
                        normalized_in_env=norm_flag,
                        recommended_norm=_rec_norm(ctrl_l, "pos", norm_flag),
                        rot_format=None,
                    ),
                )
                if end_rot > end_pos:
                    comps.append(
                        Component(
                            key=_path_for(side, "eef_rot"),
                            start=end_pos,
                            stop=end_rot,
                            width=end_rot - end_pos,
                            controller=ctrl_name,
                            side=side,
                            kind="rot",
                            normalized_in_env=norm_flag,
                            recommended_norm=_rec_norm(ctrl_l, "rot", norm_flag),
                            rot_format="axis_angle",
                        ),
                    )
                if stop > end_rot:
                    comps.append(
                        Component(
                            key=_path_for(side, "extra"),
                            start=end_rot,
                            stop=stop,
                            width=stop - end_rot,
                            controller=ctrl_name,
                            side=side,
                            kind="generic",
                            normalized_in_env=norm_flag,
                            recommended_norm=_rec_norm(ctrl_l, "generic", norm_flag),
                            rot_format=None,
                        ),
                    )
        elif role == "gripper":
            comps.append(
                Component(
                    key=_path_for(side, "gripper"),
                    start=start,
                    stop=stop,
                    width=stop - start,
                    controller=ctrl_name,
                    side=side,
                    kind="gripper",
                    normalized_in_env=norm_flag,
                    recommended_norm=_rec_norm(ctrl_l, "gripper", norm_flag),
                    rot_format=None,
                ),
            )
        else:
            # generic segment
            leaf = name_raw.replace(".", "_")
            comps.append(
                Component(
                    key=_path_for(None, leaf),
                    start=start,
                    stop=stop,
                    width=stop - start,
                    controller=ctrl_name or ctrl_typ,
                    side=None,
                    kind="generic",
                    normalized_in_env=norm_flag,
                    recommended_norm=_rec_norm(ctrl_l, "generic", norm_flag),
                    rot_format=None,
                ),
            )
    return comps


# ------------------------- running stats -------------------------


@dataclass
class RunningStats:
    width: int
    count: int = 0
    mean: np.ndarray | None = None
    M2: np.ndarray | None = None
    minv: np.ndarray | None = None
    maxv: np.ndarray | None = None

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(self.width)
        if self.mean is None:
            self.mean = np.zeros(self.width, dtype=np.float64)
            self.M2 = np.zeros(self.width, dtype=np.float64)
            self.minv = np.full(self.width, np.inf, dtype=np.float64)
            self.maxv = np.full(self.width, -np.inf, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.minv = np.minimum(self.minv, x)
        self.maxv = np.maximum(self.maxv, x)

    def to_dict(self) -> dict:
        if self.count == 0:
            z = [0.0] * self.width
            return {"count": 0, "min": z, "max": z, "mean": z, "std": z}
        var = self.M2 / max(1, self.count)
        std = np.sqrt(np.maximum(var, 1e-12))
        return {
            "count": int(self.count),
            "min": self.minv.tolist(),
            "max": self.maxv.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }


# ------------------------- HDF5 writer -------------------------


class EpisodeWriter:
    """Per-demo writer:
    - data/demo_k/actions
    - data/demo_k/action/<flat_component>
    - rewards, dones, states
    - obs/<key> and next_obs/<key> (images optional)
    """

    def __init__(
        self,
        parent_group: h5py.Group,
        demo_name: str,
        *,
        total_dim: int,
        components: list[Component],
        comp_stats: dict[str, RunningStats],
        chunk_frames: int = 128,
        compression: str | None = None,
        compression_opts: int | None = None,
        no_images: bool = False,
    ):
        self.grp = parent_group.create_group(demo_name)
        self.pos = 0
        self.total_dim = int(total_dim)
        self.components = components
        self.comp_stats = comp_stats

        self.chunk = int(max(1, chunk_frames))
        self.compression = compression
        self.compression_opts = compression_opts
        self.no_images = bool(no_images)

        # datasets
        self.actions_ds: h5py.Dataset | None = None
        self.rewards_ds: h5py.Dataset | None = None
        self.dones_ds: h5py.Dataset | None = None

        self.action_dsets: dict[str, h5py.Dataset] = {}
        self.buf_action_comp: dict[str, list[np.ndarray]] = {
            c.key: [] for c in components
        }

        self.obs_dsets: dict[str, h5py.Dataset] = {}
        self.next_obs_dsets: dict[str, h5py.Dataset] = {}
        self.last_frame: dict[str, np.ndarray] = {}
        self.buf_obs: dict[str, list[np.ndarray]] = {}

        self.buf_actions_concat: list[np.ndarray] = []
        self.buf_rewards: list[float] = []
        self.buf_dones: list[bool] = []

    def _ensure_actions_ds(self) -> h5py.Dataset:
        if self.actions_ds is not None:
            return self.actions_ds
        dim = self.total_dim
        self.actions_ds = self.grp.create_dataset(
            "actions",
            shape=(0, dim),
            maxshape=(None, dim),
            chunks=(min(self.chunk, max(1, self.chunk)), dim),
            dtype=np.float32,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression else None,
        )
        return self.actions_ds

    def _ensure_action_comp_ds(self, full_key: str, width: int) -> h5py.Dataset:
        # full_key is 'action/<flat_name>'. Create group 'action' and dataset <flat_name>.
        parts = full_key.split("/")
        g = self.grp
        for p in parts[:-1]:
            g = g.require_group(p)
        name = parts[-1]
        if name in g:
            return g[name]  # type: ignore
        return g.create_dataset(
            name,
            shape=(0, width),
            maxshape=(None, width),
            chunks=(min(self.chunk, max(1, self.chunk)), width),
            dtype=np.float32,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression else None,
        )

    def _ensure_obs_ds(self, key: str, proto: np.ndarray):
        if key in self.obs_dsets:
            return
        arr = obs_storage_array(proto)
        if is_image_like(arr):
            arr = ensure_uint8_hwc(arr)
            if arr.ndim == 4:
                arr = arr[0]
        elif arr.ndim == 0:
            arr = arr.reshape(1)
        ch = (min(self.chunk, 64), *arr.shape)
        self.obs_dsets[key] = self.grp.create_dataset(
            f"obs/{key}",
            shape=(0, *arr.shape),
            maxshape=(None, *arr.shape),
            chunks=ch,
            dtype=arr.dtype,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression else None,
        )
        self.next_obs_dsets[key] = self.grp.create_dataset(
            f"next_obs/{key}",
            shape=(0, *arr.shape),
            maxshape=(None, *arr.shape),
            chunks=ch,
            dtype=arr.dtype,
            compression=self.compression,
            compression_opts=self.compression_opts if self.compression else None,
        )
        self.buf_obs.setdefault(key, [])

    def append(self, item: dict[str, Any]):
        # actions
        if "action" in item:
            a = to_np1(item["action"], np.float32).reshape(-1)
            if a.size != self.total_dim:
                if a.size < self.total_dim:
                    pad = np.zeros((self.total_dim - a.size,), dtype=np.float32)
                    a = np.concatenate([a, pad], axis=0)
                else:
                    a = a[: self.total_dim]
            self.buf_actions_concat.append(a)
            for c in self.components:
                seg = a[c.start : c.stop].astype(np.float32, copy=False)
                self.buf_action_comp[c.key].append(seg)
                self.comp_stats[c.key].update(seg)

        # rewards
        if "reward" in item:
            r = to_np1(item["reward"], np.float32)
            if self.rewards_ds is None:
                self.rewards_ds = self.grp.create_dataset(
                    "rewards",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(min(self.chunk, 1024),),
                    dtype=np.float32,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                    if self.compression
                    else None,
                )
            self.buf_rewards.append(float(r.reshape(-1)[0]) if r.size else 0.0)

        # dones
        s = (
            bool(to_np1(item.get("success", False), np.bool_).reshape(-1)[0])
            if "success" in item
            else False
        )
        t = (
            bool(to_np1(item.get("truncated", False), np.bool_).reshape(-1)[0])
            if "truncated" in item
            else False
        )
        self.buf_dones.append(s or t)

        # obs
        for k, v in item.items():
            if not str(k).startswith("obs."):
                continue
            if not is_numeric_array(v):
                continue
            ok = obs_key_clean(k)
            img_like = is_image_like(v)
            if self.no_images and img_like:
                continue
            self._ensure_obs_ds(ok, v)
            arr = obs_storage_array(v)
            if img_like:
                arr = ensure_uint8_hwc(arr)
                if arr.ndim == 4:
                    arr = arr[0]
            elif arr.ndim == 0:
                arr = arr.reshape(1)
            self.buf_obs.setdefault(ok, []).append(arr)

        # flush on block
        if len(self.buf_dones) >= self.chunk:
            self.flush()

    def flush(self):
        batch_size = len(self.buf_dones)
        if batch_size == 0:
            return
        start, end = self.pos, self.pos + batch_size

        # dones
        if self.dones_ds is None:
            self.dones_ds = self.grp.create_dataset(
                "dones",
                shape=(0,),
                maxshape=(None,),
                chunks=(min(self.chunk, 1024),),
                dtype=np.bool_,
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression else None,
            )
        self.dones_ds.resize(end, axis=0)
        self.dones_ds[start:end] = np.asarray(self.buf_dones, dtype=np.bool_)

        # rewards
        if self.rewards_ds is not None and self.buf_rewards:
            self.rewards_ds.resize(end, axis=0)
            self.rewards_ds[start:end] = np.asarray(self.buf_rewards, dtype=np.float32)

        # flat actions
        if self.buf_actions_concat:
            ds = self._ensure_actions_ds()
            arr = np.stack(self.buf_actions_concat, axis=0)  # (B, D)
            ds.resize(end, axis=0)
            ds[start:end] = arr

        # components
        for c in self.components:
            frames = self.buf_action_comp[c.key]
            if not frames:
                continue
            arr = np.stack(frames, axis=0)
            ds = self.action_dsets.get(c.key)
            if ds is None:
                ds = self._ensure_action_comp_ds(c.key, c.width)
                self.action_dsets[c.key] = ds
            ds.resize(end, axis=0)
            ds[start:end] = arr

        # obs + next_obs
        for ok, frames in self.buf_obs.items():
            if not frames:
                continue
            arr = np.stack(frames, axis=0)
            ds = self.obs_dsets[ok]
            ds.resize(end, axis=0)
            ds[start:end] = arr

            dsn = self.next_obs_dsets[ok]
            if batch_size > 1:
                dsn.resize(end - 1, axis=0)
                dsn[start : end - 1] = arr[1:]
            if self.pos > 0 and ok in self.last_frame:
                dsn.resize(end, axis=0)
                dsn[start - 1] = arr[0]
            self.last_frame[ok] = arr[-1]

        self.pos = end
        self.buf_actions_concat.clear()
        self.buf_rewards.clear()
        self.buf_dones.clear()
        for k in list(self.buf_action_comp.keys()):
            self.buf_action_comp[k].clear()
        for k in list(self.buf_obs.keys()):
            self.buf_obs[k].clear()

    def finalize(self) -> int:
        self.flush()
        length = self.pos
        for ok, last in self.last_frame.items():
            dsn = self.next_obs_dsets[ok]
            if dsn.shape[0] < length:
                dsn.resize(length, axis=0)
            dsn[length - 1] = last
        self.grp.create_dataset("states", shape=(length, 0))
        self.grp.attrs["num_samples"] = length
        return length


# ------------------------- dataset + fallback -------------------------


def _build_dataset_with_fallback(
    repo_or_root: str,
    preferred: str | None,
    allow_fallback: bool,
    download_videos: bool,
) -> tuple[LeRobotDataset, str | None]:
    tried: list[str | None] = []
    order: list[str | None]
    if preferred in (None, "auto"):
        order = [None, *_NON_TC_BACKENDS]
    else:
        order = [preferred] + [be for be in _NON_TC_BACKENDS if be != preferred]

    last_err = None
    for be in order:
        try:
            kwargs = {"download_videos": download_videos}
            if be is not None:
                kwargs["video_backend"] = be
            if is_hf_repo_id(repo_or_root):
                ds = LeRobotDataset(repo_or_root, **kwargs)
            else:
                root = Path(repo_or_root).resolve()
                ds = LeRobotDataset(
                    repo_id=repo_id_from_local_root(root),
                    root=str(root),
                    **kwargs,
                )

            # lightly validate
            _ = ds.hf_dataset[0] if not download_videos else ds[0]
            print(f"[backend] Using video backend: {'auto' if be is None else be}")
            return ds, be
        except Exception as e:
            tried.append("auto" if be is None else be)
            last_err = e
            if not allow_fallback:
                break
    raise RuntimeError(f"All backends failed (tried={tried}). Last error: {last_err}")


# ------------------------- file-level metadata + external JSON -------------------------


def _write_file_attrs(
    f: h5py.File,
    *,
    repo_or_root: str,
    sample_keys: list[str],
    env_config: dict[str, Any],
    action_layout: dict[str, Any],
    components: list[Component],
    stats_map: dict[str, RunningStats],
):
    f.attrs["repo_id"] = repo_or_root
    f.attrs["sample_keys"] = json.dumps(sample_keys)
    f.attrs["action_layout"] = json.dumps(action_layout)

    action_keys = [c.key for c in components]
    action_slices = {c.key: [int(c.start), int(c.stop)] for c in components}
    action_config = {
        c.key: {
            "controller": c.controller,
            "normalized_in_env": bool(c.normalized_in_env),
            "recommended_normalization": c.recommended_norm,
            "rot_format": c.rot_format,
        }
        for c in components
    }
    f.attrs["robomimic_action"] = json.dumps(
        {
            "action_keys": action_keys,
            "action_slices": action_slices,
            "action_config": action_config,
        },
    )
    f.attrs["action_components_stats"] = json.dumps(
        {c.key: stats_map[c.key].to_dict() for c in components},
    )

    g = f.require_group("data")
    g.attrs["env_args"] = json.dumps(
        {
            "env_name": env_config.get("task"),
            "type": 4,
            "env_kwargs": env_config,
        },
    )


def _write_external_action_config_json(
    out_dir: Path,
    *,
    repo_or_root: str,
    env_config: dict[str, Any],
    action_layout: dict[str, Any],
    components: list[Component],
):
    """Produce action_config.json next to the HDF5, grouping components by original segments."""
    task = env_config.get("task")
    comps_by_range: dict[tuple[int, int], list[Component]] = {}
    for c in components:
        comps_by_range.setdefault((c.start, c.stop), []).append(c)

    segs = sorted(
        action_layout.get("segments", []),
        key=lambda s: int(s.get("start", 0)),
    )
    robomimic_action_config = {
        c.key: {
            "normalization": None
            if c.normalized_in_env
            else None
            if c.recommended_norm == "none"
            else c.recommended_norm,
        }
        for c in components
    }

    seg_out: list[dict[str, Any]] = []
    for seg in segs:
        name_raw = str(seg.get("name", "segment"))
        start = int(seg.get("start", 0))
        stop = int(seg.get("stop", start))
        width = int(seg.get("width", stop - start))
        ctrl_type = str(seg.get("ctrl_type", ""))

        # pick controller + normalized flag from env_config (if role known)
        side, role = _side_role(name_raw)
        if role in ("manipulator", "gripper"):
            ctrl_name, norm_flag = _ctrl_info(env_config, role, side)
        else:
            ctrl_name, norm_flag = (ctrl_type or "UnknownController", False)

        # collect parts: all components that live fully inside this seg
        parts = [
            {
                "name": "eef_pos"
                if c.kind == "pos"
                else "eef_rot"
                if c.kind == "rot"
                else "eef_quat"
                if c.kind == "quat"
                else "gripper"
                if c.kind == "gripper"
                else "extra",
                "robomimic_id": c.key,
                "slice": [int(c.start), int(c.stop)],
                "dim": int(c.width),
                "normalized_in_env": bool(c.normalized_in_env),
                "normalization": robomimic_action_config[c.key]["normalization"],
                "recommended_normalization": c.recommended_norm,
                "rot_format": c.rot_format,
            }
            for c in components
            if c.start >= start and c.stop <= stop
        ]

        seg_out.append(
            {
                "name": name_raw,
                "start": start,
                "stop": stop,
                "width": width,
                "ctrl_type": ctrl_type or ctrl_name,
                "controller": ctrl_name,
                "normalized": bool(norm_flag),
                "parts": parts,
            },
        )

    payload = {
        "repo_id": repo_or_root,
        "task": task,
        "total_dim": int(action_layout.get("total_dim", 0)),
        "action_keys": [c.key for c in components],
        "action_slices": {c.key: [int(c.start), int(c.stop)] for c in components},
        "robomimic_action_config": robomimic_action_config,
        "segments": seg_out,
        "notes": "This file mirrors meta/action_layout.json and adds executable robomimic action mapping.",
    }

    with (out_dir / "action_config.json").open("w") as f:
        json.dump(payload, f, indent=2)


# ------------------------- main conversion -------------------------


def convert_lerobot_to_hdf5(
    repo_or_root: str,
    output_path: str,
    split: str = "train",
    *,
    h5_basename: str
    | None = "dataset",  # <-- set by caller to repo name, e.g. "lift_ur5_robotiq"
    video_backend: str | None = "auto",
    allow_fallback: bool = True,
    read_workers: int = 8,
    block_size: int = 128,
    compression: str | None = None,
    compression_opts: int | None = None,
    no_images: bool = False,
) -> None:
    _ = split
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # choose output h5 name
    base = (h5_basename or "dataset").strip()
    out_h5 = out_dir / f"{base}.hdf5"

    env_config = load_env_config(repo_or_root)
    action_layout = load_action_layout(repo_or_root)
    total_dim = int(action_layout.get("total_dim", 0))
    if total_dim <= 0:
        raise ValueError("action_layout.total_dim must be > 0")

    components = build_components(action_layout, env_config)
    stats_map: dict[str, RunningStats] = {
        c.key: RunningStats(c.width) for c in components
    }

    local_root = _local_lerobot_root(repo_or_root)
    if no_images and local_root is not None:
        columns = _local_parquet_columns(local_root, no_images=True)
        sample_keys = sorted(columns)
        it = _iter_local_parquet_rows(
            local_root,
            columns=columns,
            batch_size=block_size,
        )
        used_be = "local-parquet"
        print("[convert] Streaming local parquet rows without LeRobotDataset.")
    else:
        ds, used_be = _build_dataset_with_fallback(
            repo_or_root,
            preferred=None if (video_backend in (None, "auto")) else video_backend,
            allow_fallback=allow_fallback,
            download_videos=not no_images,
        )

        # preview keys
        try:
            sample0 = ds.hf_dataset[0] if no_images else ds[0]
            sample_keys = sorted(sample0.keys())
        except Exception:
            sample_keys = []

        # iterate (with prefetch)
        indices = list(range(len(ds.hf_dataset) if no_images else len(ds)))
        prefetch = max(block_size, read_workers * 2)
        if read_workers <= 1:

            def _iter():
                for i in indices:
                    yield i, (ds.hf_dataset[i] if no_images else ds[i])

            it = _iter()
        else:
            import concurrent.futures as fut

            def _iter():
                with fut.ThreadPoolExecutor(max_workers=read_workers) as ex:
                    it_idx = iter(indices)
                    inflight: dict[int, fut.Future] = {}
                    getter = ds.hf_dataset.__getitem__ if no_images else ds.__getitem__
                    for _ in range(min(prefetch, len(indices))):
                        i = next(it_idx, None)
                        if i is None:
                            break
                        inflight[i] = ex.submit(getter, i)
                    next_i = indices[0] if indices else None
                    while inflight:
                        futr = inflight.pop(next_i)
                        yield next_i, futr.result()
                        j = next(it_idx, None)
                        if j is not None:
                            inflight[j] = ex.submit(getter, j)
                        next_i += 1
                        while next_i not in inflight and inflight:
                            next_i += 1

            it = _iter()

    # written under a temporary name and moved into place at the end: an interrupted conversion never leaves a
    # truncated HDF5 under the final name (nor replaces a good one)
    tmp_h5 = out_h5.with_name(out_h5.name + ".tmp")
    with h5py.File(tmp_h5, "w") as f:
        _write_file_attrs(
            f,
            repo_or_root=repo_or_root,
            sample_keys=sample_keys,
            env_config=env_config,
            action_layout=action_layout,
            components=components,
            stats_map=stats_map,
        )

        data_grp = f.require_group("data")
        total_steps = 0
        num_demos = 0

        current_ep = None
        writer: EpisodeWriter | None = None

        for idx, item in it:
            if "episode_index" not in item:
                raise KeyError(
                    f"Sample {idx} missing 'episode_index'. Keys: {sorted(item.keys())[:30]}",
                )
            ei = item["episode_index"]
            ei = int(ei.item()) if isinstance(ei, torch.Tensor) else int(ei)

            if current_ep is None:
                current_ep = ei
                writer = EpisodeWriter(
                    data_grp,
                    f"demo_{num_demos}",
                    total_dim=total_dim,
                    components=components,
                    comp_stats=stats_map,
                    chunk_frames=block_size,
                    compression=compression,
                    compression_opts=compression_opts,
                    no_images=no_images,
                )
            elif ei != current_ep:
                steps = writer.finalize()  # type: ignore
                total_steps += steps
                num_demos += 1
                current_ep = ei
                writer = EpisodeWriter(
                    data_grp,
                    f"demo_{num_demos}",
                    total_dim=total_dim,
                    components=components,
                    comp_stats=stats_map,
                    chunk_frames=block_size,
                    compression=compression,
                    compression_opts=compression_opts,
                    no_images=no_images,
                )

            # pass-through sample
            sample = {}
            if "action" in item:
                sample["action"] = item["action"]
            if "reward" in item:
                sample["reward"] = item["reward"]
            if "success" in item:
                sample["success"] = item["success"]
            if "truncated" in item:
                sample["truncated"] = item["truncated"]
            elif "failure" in item:
                with suppress(Exception):
                    sample["success"] = (~item["failure"].bool()).to(torch.bool)
            sample.update(
                {k: v for k, v in item.items() if str(k).startswith("obs.")},
            )

            writer.append(sample)  # type: ignore

        if writer is not None:
            steps = writer.finalize()
            total_steps += steps
            num_demos += 1

        f.attrs["num_demos"] = num_demos
        data_grp.attrs["total"] = total_steps
        f.create_group("mask")

    # external JSON (next to HDF5)
    _write_external_action_config_json(
        out_dir,
        repo_or_root=repo_or_root,
        env_config=env_config,
        action_layout=action_layout,
        components=components,
    )

    tmp_h5.replace(out_h5)  # the HDF5 appears only once it is complete
    tag = "(no-images)" if no_images else ""
    print(
        f"[convert] Wrote HDF5 {tag} -> {out_h5} | demos={num_demos} | frames={total_steps} | backend={'auto' if used_be is None else used_be}",
    )
