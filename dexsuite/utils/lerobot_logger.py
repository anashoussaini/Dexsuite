"""Write DexSuite rollouts to LeRobot datasets.

Frames are flattened into LeRobot feature dictionaries and image observations are kept as uint8 arrays. Dataset metadata mirrors the environment options and action layout so recorded runs can be interpreted offline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.datasets.io_utils import write_info
except ModuleNotFoundError:  # LeRobot >= 0.4
    from lerobot.datasets.utils import write_info

from dexsuite.utils.options_utils import get_ds_converter

DEFAULT_DATASET_ROOT = Path("collected_dataset")


class _ScalarCompatibleDataset(LeRobotDataset):
    """Preserve LeRobot features while giving Arrow genuine scalar columns.

    LeRobot 0.4.4 stacks shape-(1,) numeric features to (T,1), but its
    HuggingFace feature is Value, not a sequence. NumPy 2.4 correctly rejects
    float(array([x])). Normalize only those scalar columns at the Arrow boundary;
    statistics and add_frame validation retain the original feature shapes.
    """
    def _save_episode_data(self, episode_buffer):
        import datasets
        compatible = dict(episode_buffer)
        for key, feature in self.hf_features.items():
            if isinstance(feature, datasets.Value):
                value = np.asarray(compatible[key])
                if value.ndim == 2 and value.shape[1] == 1:
                    compatible[key] = value[:, 0]
        return super()._save_episode_data(compatible)


class LeRobotLogger:
    """Record environment rollouts into a LeRobotDataset on disk."""

    @staticmethod
    def _arm_repo_part(arm_options) -> str:
        part = str(arm_options.manipulator)
        if arm_options.gripper is not None:
            part = f"{part}_{arm_options.gripper}"
        return part

    def __init__(self, task, env, root_dir=None, dataset_dir=None, extra_features=None):
        """Initialize the object.

        Args:
        task: Task name used in the dataset repo ID and output folder structure.
        env: DexSuite environment instance that provides spaces and options metadata.
        root_dir: Root directory to store datasets under. Defaults to None.
        dataset_dir: Exact dataset directory that overrides the timestamp layout. Defaults to None.
        """
        self.env = env
        self.schema = self.extract_scheme(
            self.env.observation_space,
            self.env.action_space,
            parent_key="obs",
            rgb_as="video",
        )

        self.extra_features = dict(extra_features or {})
        overlap = self.schema.keys() & self.extra_features.keys()
        if overlap:
            raise ValueError(f"Extra features overwrite native fields: {sorted(overlap)}")
        self.schema.update(self.extra_features)
        self.sim = self.env.sim_options
        self.robot_options = self.env.robot_options
        self.cameras_options = self.env.cameras_options
        self.task = task
        self.fps = int(self.sim.control_hz)
        self.repo_id = ""

        if self.robot_options.type_of_robot == "single":
            arm = self._arm_repo_part(self.robot_options.single)
            self.repo_id = f"lerobot/dexsuite_{self.task}_{arm}"
        elif self.robot_options.type_of_robot == "bimanual":
            left = self._arm_repo_part(self.robot_options.left)
            right = self._arm_repo_part(self.robot_options.right)
            self.repo_id = f"lerobot/dexsuite_{self.task}_left_{left}_right_{right}"

        timestamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
        if dataset_dir is not None:
            self.dataset_dir = Path(dataset_dir)
        elif root_dir is None:
            self.dataset_dir = (
                DEFAULT_DATASET_ROOT / self.task / f"{self.task}_{timestamp}"
            )
        else:
            self.dataset_dir = Path(root_dir) / self.task / f"{self.task}_{timestamp}"
        info_path = self.dataset_dir / "meta" / "info.json"
        if info_path.exists():
            saved = json.loads(info_path.read_text())
            features = saved["features"]
            mismatched = [key for key, spec in self.schema.items()
                          if key not in features or features[key]["dtype"] != spec["dtype"]
                          or tuple(features[key]["shape"]) != tuple(spec["shape"])]
            if saved["fps"] != self.fps or mismatched:
                raise ValueError(f"Existing LeRobot dataset schema/fps differs; use a new directory. Fields: {mismatched}")
            self.ds = _ScalarCompatibleDataset(
                repo_id=self.repo_id,
                root=self.dataset_dir,
                download_videos=True,
            )
        else:
            self.ds = _ScalarCompatibleDataset.create(
                repo_id=self.repo_id,
                root=self.dataset_dir,
                fps=self.fps,
                features=self.schema,
                use_videos=True,
            )
        self._metadata()

    def add_step_pre(self, obs_before, action, reward, terminated, truncated, info, extra_frame=None):
        """Add one transition to the current episode buffer.

        Args:
            obs_before: Observation dictionary before the action is applied.
            action: Action tensor to validate or dispatch.
            reward: Reward value for the transition.
            terminated: Termination flag for the transition.
            truncated: Truncation flag for the transition.
            info: Environment info dictionary for the transition.
        """
        frame = self._construct_frame(
            obs_before,
            reward,
            terminated,
            truncated,
            info,
            action,
        )
        extra_frame = dict(extra_frame or {})
        if extra_frame.keys() != self.extra_features.keys():
            raise ValueError("Extra frame keys must match declared extra features")
        frame.update(extra_frame)
        # Double-precision physics may return float64 observations while the
        # environment's public observation space still declares float32.
        for key, feature in self.schema.items():
            if feature["dtype"].startswith("float") and key in frame:
                value = np.asarray(frame[key])
                if np.issubdtype(value.dtype, np.floating):
                    value = value.astype(feature["dtype"], copy=False)
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite feature after schema conversion: {key}")
                    frame[key] = value
        self.ds.add_frame(frame)

    def save_episode(self):
        """Finalize and save the current episode to the dataset."""
        self.ds.save_episode()

    def clear_buffer(self):
        # lerobot clear_episode_buffer() only deletes buffered frames for
        # meta.image_keys; VIDEO keys buffered pngs survive and get encoded
        # into the NEXT saved episode video (dropped attempts leaked into
        # episode 0 on 2026-09-03). Delete them ourselves.
        import shutil
        self.ds._wait_image_writer()
        ep = self.ds.episode_buffer["episode_index"]
        ep = int(ep if not hasattr(ep, "__len__") else ep[0])
        for key in self.ds.meta.video_keys:
            img_dir = self.ds._get_image_file_dir(ep, key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir, ignore_errors=True)
        """Clear the current episode buffer without saving."""
        self.ds.clear_episode_buffer()

    def close(self):
        """Close the logger and flush dataset state."""
        if self.ds.episode_buffer and self.ds.episode_buffer.get("size", 0):
            self.clear_buffer()
        self.ds.finalize()

    def extract_scheme(
        self,
        obs_space: gym.spaces.Dict,
        action_space: gym.spaces.Space | None = None,
        *,
        parent_key: str = "obs",
        add_action_feature: bool = False,
        action_key: str = "obs.state.other.action",
        rgb_as: str = "video",
    ) -> dict[str, dict]:
        """Build the LeRobot feature schema from an observation space.

        Args:
            obs_space: Gymnasium Dict observation space.
            action_space: Optional action space (used to populate the "action" feature).
            parent_key: Prefix for nested observation keys.
            add_action_feature: Deprecated (kept for API compatibility).
            action_key: Deprecated (kept for API compatibility).
            rgb_as: How to encode RGB observations (currently always uses "video").

        Returns:
            dict[str, dict]: Mapping of flattened feature names to dtype/shape metadata.
        """
        del add_action_feature, action_key
        schema: dict[str, dict] = {}
        for key, val in obs_space.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(val, gym.spaces.Dict):
                schema.update(
                    self.extract_scheme(
                        val,
                        action_space,
                        parent_key=new_key,
                        rgb_as=rgb_as,
                    ),
                )
            elif isinstance(val, gym.spaces.Box):
                shape = val.shape if val.shape != () else (1,)
                if new_key.startswith("obs.cameras.") and new_key.endswith(".rgb"):
                    schema[new_key] = {"dtype": "video", "shape": shape}
                else:
                    schema[new_key] = {"dtype": str(val.dtype), "shape": shape}
        schema.update(
            {
                "reward": {"dtype": "float32", "shape": (1,)},
                "truncated": {"dtype": "bool", "shape": (1,)},
                "terminated": {"dtype": "bool", "shape": (1,)},
                "success": {"dtype": "bool", "shape": (1,)},
                "failure": {"dtype": "bool", "shape": (1,)},
            },
        )
        act_shape = action_space.shape
        schema["action"] = {"dtype": "float32", "shape": act_shape}
        return schema

    def _construct_frame(
        self,
        obs,
        reward,
        terminated,
        truncated,
        info,
        action,
    ):
        """Flatten a step into the LeRobot frame dict expected by LeRobotDataset."""
        frame = {}

        def flatten(o, prefix="obs"):
            for k, v in o.items():
                nk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, nk)
                else:
                    frame[nk] = (
                        v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    )

        flatten(obs)

        def to_np1(x, dtype):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.shape == ():
                x = x.reshape(
                    1,
                )
            return x.astype(dtype, copy=False)

        frame["reward"] = to_np1(reward, np.float32)
        frame["success"] = to_np1(info.get("success", False), np.bool_)
        frame["failure"] = to_np1(info.get("failure", False), np.bool_)
        frame["truncated"] = to_np1(truncated, np.bool_)
        frame["terminated"] = to_np1(terminated, np.bool_)
        frame["action"] = to_np1(action, np.float32)
        frame["task"] = self.task
        return frame

    def _metadata(self):
        """Write dataset metadata files (options + action layout)."""
        info = self.ds.meta.info
        self._save_layout_json()
        info["task"] = self.task
        info["options"] = self._save_options_json()
        write_info(info, self.ds.meta.root)

    def _save_layout_json(self):
        """Persist the action layout (segment breakdown) to meta/action_layout.json."""
        layout = self.env.robot._layout
        meta_dir = self.dataset_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_dim": layout.total_dim,
            "segments": [
                {
                    "name": ".".join(seg.name),
                    "start": int(seg.start),
                    "stop": int(seg.stop),
                    "width": int(seg.width()),
                    "ctrl_type": type(seg.ctrl).__name__,
                }
                for seg in layout.segments
            ],
        }
        with (meta_dir / "action_layout.json").open("w") as f:
            json.dump(payload, f, indent=2)

    def _save_options_json(self):
        """Persist the environment options to meta/env_config.json."""
        meta_dir = self.dataset_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        c = get_ds_converter()
        blob = {}
        blob["task"] = self.task
        blob["options"] = {
            "sim": c.unstructure(self.sim),
            "robot": c.unstructure(self.robot_options),
            "cameras": c.unstructure(self.cameras_options),
        }
        # Freeze resolved mount/intrinsic settings: old datasets must not depend
        # on a future version of cameras.yaml during conversion or evaluation.
        if hasattr(self.env, "camera_sys"):
            dynamic = blob["options"]["cameras"]["dynamic"]
            for name, (_, spec, _, _) in self.env.camera_sys._dyn_specs.items():
                dynamic[name] = c.unstructure(spec)
        with (meta_dir / "env_config.json").open("w") as f:
            json.dump(blob, f, indent=2)
