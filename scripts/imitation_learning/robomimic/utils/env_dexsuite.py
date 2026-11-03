"""
This file contains the dexsuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""

from copy import deepcopy
from typing import Any

import dexsuite as ds
import numpy as np
import torch
from dexsuite.options import (
    AABBOptions,
    ArmOptions,
    CamerasOptions,
    ControllerOptions,
    DynamicCamOptions,
    LayoutOptions,
    PoseOptions,
    RobotOptions,
    SimOptions,
    StaticCamOptions,
)
from dexsuite.utils import get_device

import robomimic.envs.env_base as EB

# Stock robomimic only defines types 1-3; DexSuite datasets are stamped with 4.
if not hasattr(EB.EnvType, "DEXSUITE_TYPE"):
    EB.EnvType.DEXSUITE_TYPE = 4


def _t2(x):
    return (int(x[0]), int(x[1]))


def _t3(x):
    return (float(x[0]), float(x[1]), float(x[2]))


def _t4(x):
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))


def _ctrl(d: dict) -> ControllerOptions:
    return ControllerOptions(name=d["name"], config=dict(d.get("config", {})))


def _aabb(d: dict) -> AABBOptions:
    return AABBOptions(min=_t3(d["min"]), max=_t3(d["max"]))


def _arm(d: dict) -> ArmOptions:
    return ArmOptions(
        manipulator=d["manipulator"],
        gripper=d.get("gripper"),
        manipulator_controller=_ctrl(d["manipulator_controller"]),
        gripper_controller=(
            _ctrl(d["gripper_controller"]) if d.get("gripper_controller") else None
        ),
        workspace=(_aabb(d["workspace"]) if d.get("workspace") else None),
    )


def _pose(d: dict | None) -> PoseOptions | None:
    if d is None:
        return None
    return PoseOptions(
        pos=_t3(d["pos"]),
        yaw_rad=float(d.get("yaw_rad", 0.0)),
        quat=(None if d.get("quat") is None else _t4(d["quat"])),
    )


def _layout(d: dict | None) -> LayoutOptions:
    if d is None:
        return LayoutOptions()
    return LayoutOptions(
        preset=d.get("preset"),
        params=dict(d.get("params", {})),
        single=_pose(d.get("single")),
        left=_pose(d.get("left")),
        right=_pose(d.get("right")),
    )


def _cams(d: dict) -> CamerasOptions:
    static = {
        name: (
            val
            if isinstance(val, str)
            else StaticCamOptions(
                pos=_t3(val["pos"]),
                lookat=_t3(val["lookat"]),
                fov=float(val.get("fov", 55.0)),
                res=_t2(val.get("res", (224, 224))),
            )
        )
        for name, val in (d.get("static") or {}).items()
    }

    dynamic = {
        name: (
            val
            if isinstance(val, str)
            else DynamicCamOptions(
                link=val.get("link"),
                pos_offset=_t3(val.get("pos_offset", (0.0, 0.0, 0.0))),
                quat_offset=_t4(val.get("quat_offset", (1.0, 0.0, 0.0, 0.0))),
                fov=float(val.get("fov", 60.0)),
                res=_t2(val.get("res", (224, 224))),
            )
        )
        for name, val in (d.get("dynamic") or {}).items()
    }

    return CamerasOptions(
        static=static, dynamic=dynamic, modalities=tuple(d.get("modalities", ("rgb",)))
    )


def _extract_env_options(blob: dict) -> tuple[SimOptions, RobotOptions, CamerasOptions]:
    opts = blob.get("options", blob)
    sim = SimOptions(
        control_hz=int(opts["sim"]["control_hz"]),
        performance_mode=bool(opts["sim"]["performance_mode"]),
        n_envs=int(opts["sim"]["n_envs"]),
    )
    r = opts["robot"]
    robot = RobotOptions(
        type_of_robot=r["type_of_robot"],
        single=_arm(r["single"]) if r.get("single") else None,
        left=_arm(r["left"]) if r.get("left") else None,
        right=_arm(r["right"]) if r.get("right") else None,
        layout=_layout(r.get("layout")),
        visualize_tcp=bool(r.get("visualize_tcp", False)),
        visualize_aabb=bool(r.get("visualize_aabb", False)),
    )
    cameras = _cams(opts["cameras"])
    return sim, robot, cameras


def _to_numpy(x: Any):
    """torch.Tensor -> numpy; leave numpy as is."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _to_chw01(img: np.ndarray) -> np.ndarray:
    """
    Camera frames: convert to CHW float32 in [0,1]. No other magic.
    Accept HWC uint8 / float or CHW already.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[-1] in (1, 3, 4):
        a = np.transpose(a, (2, 0, 1))
    elif a.ndim == 3 and a.shape[0] in (1, 3, 4):
        pass
    else:
        raise ValueError(
            f"Unexpected image shape {a.shape} (expect HxW, HxWxC, or CxHxW)"
        )

    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32, copy=False)

    return a


def _flatten_obs_dict(obs):
    """
    obs: nested dict of torch.Tensors from Dexsuite:
         { 'state': {...}, 'cameras': {'front': {'rgb': (H,W,C)}, ... } }
    Returns a flat dict with underscore-joined keys, numpy arrays.
    This function ensures that RGB images are converted to HWC format,
    as expected by robomimic's default processors.
    """
    out = {}

    def to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def is_hwc_image(a):
        return (a.ndim == 3) and (a.shape[2] in (1, 3, 4))

    def is_chw_image(a):
        return (a.ndim == 3) and (a.shape[0] in (1, 3, 4))

    def add(prefix, v):
        if isinstance(v, dict):
            for k, vv in v.items():
                add(f"{prefix}_{k}" if prefix else k, vv)
        else:
            a = to_np(v)
            key = prefix.replace("/", "_")

            if key.endswith("_rgb"):
                # Robomimic's default processor expects HWC.
                # We must detect the format and convert if necessary.

                if is_chw_image(a):
                    # It's CHW (3, H, W), transpose to HWC (H, W, 3)
                    a = np.transpose(a, (1, 2, 0))
                elif is_hwc_image(a):
                    # It's already HWC, do nothing
                    pass
                elif a.ndim == 2:
                    # Grayscale image, add channel dim
                    a = a[..., None]
                else:
                    raise ValueError(
                        f"Unexpected image shape {a.shape} for key {key}"
                    )

                # Now 'a' is guaranteed to be HWC (or HW1)
                if a.dtype != np.float32:
                    a = a.astype(np.float32)

                if a.max() > 1.01:
                    a = a / 255.0

                out[key] = a  # 'a' is HWC

            else:
                # Not an RGB image, just convert type and add
                if a.dtype != np.float32 and np.issubdtype(a.dtype, np.number):
                    a = a.astype(np.float32)
                out[key] = a

    for top in ("state", "cameras"):
        if top in obs:
            add(top, obs[top])

    for k, v in obs.items():
        if k not in ("state", "cameras"):
            add(k, v)

    return out


class EnvDexsuite(EB.EnvBase):
    def __init__(
        self,
        env_name: str,
        render: bool = False,
        render_offscreen: bool = False,
        use_image_obs: bool = False,
        use_depth_obs: bool = False,
        **env_config: Any,
    ):
        super().__init__(
            env_name,
            render,
            render_offscreen,
            use_image_obs,
            use_depth_obs,
            **env_config,
        )

        self._init_kwargs = deepcopy(env_config)
        self._env_name = env_name
        self._device = get_device()

        self._env_config = env_config
        self._task = env_config.get("task", env_name)
        self._sim_opts, self._robot_opts, self._cam_opts = _extract_env_options(
            env_config
        )

        self._render_mode = (
            "human"
            if render
            else "rgb_array"
            if render_offscreen or use_image_obs
            else None
        )

        self._env = ds.make(
            self._task,
            sim=self._sim_opts,
            robot=self._robot_opts,
            cameras=self._cam_opts,
            render_mode=self._render_mode,
        )

        self._action_dim = int(self._env.action_space.shape[0])

        self._last_obs_raw = None
        self._last_obs_np: dict | None = None
        self._last_reward: float | None = None
        self._last_done: bool | None = None
        self._last_success: bool = False

    def step(self, action):
        if isinstance(action, np.ndarray):
            act_t = torch.from_numpy(action).to(self._device, dtype=torch.float32)
        elif torch.is_tensor(action):
            act_t = action.to(self._device, dtype=torch.float32)
        else:
            raise TypeError(
                f"action must be np.ndarray or torch.Tensor, got {type(action)}"
            )

        obs_t, reward_t, terminated_t, truncated_t, info = self._env.step(act_t)

        self._last_obs_raw = obs_t

        self._last_reward = float(reward_t.item())
        done = bool(terminated_t.item() or truncated_t.item())
        self._last_done = done

        succ = info.get("success", False)
        self._last_success = bool(succ.item()) if torch.is_tensor(succ) else bool(succ)

        self._last_obs_np = _flatten_obs_dict(obs_t)
        return self._last_obs_np, self._last_reward, done, info

    def reset(self):
        self._last_done = None
        self._last_success = False
        self._last_reward = None
        obs_t, _ = self._env.reset()
        obs_np = _flatten_obs_dict(obs_t)
        self._last_obs_np = obs_np
        return obs_np

    def render(self, mode="human", height=None, width=None, camera_name=None):
        img = self._last_obs_np.get("cameras_front_rgb")

        if img is None:
            return None

        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if height is not None and width is not None:
            H, W = int(height), int(width)
            if (arr.shape[0], arr.shape[1]) != (H, W):
                yy = np.linspace(0, arr.shape[0] - 1, H).astype(np.int64)
                xx = np.linspace(0, arr.shape[1] - 1, W).astype(np.int64)
                arr = arr[yy][:, xx]

        return arr

    def get_observation(self, obs=None):
        if obs is None:
            if self._last_obs_np is None:
                raise RuntimeError(
                    "No cached observation; call reset() or step() first."
                )
            return self._last_obs_np
        return _flatten_obs_dict(obs)

    def get_reward(self):
        return 0.0 if self._last_reward is None else float(self._last_reward)

    def is_done(self):
        return bool(self._last_done) if self._last_done is not None else False

    def is_success(self):
        return {"task": bool(self._last_success)}

    @property
    def action_dimension(self):
        return self._action_dim

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EB.EnvType.DEXSUITE_TYPE

    @property
    def version(self):
        return "dexsuite-0"

    def serialize(self):
        return dict(
            env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs)
        )

    @property
    def rollout_exceptions(self):
        return tuple()

    @property
    def base_env(self):
        return self._env

    def reset_to(self, state):
        raise NotImplementedError("Dexsuite reset_to is not supported")

    def get_state(self):
        return {}

    def get_goal(self):
        raise NotImplementedError

    def set_goal(self, **kwargs):
        raise NotImplementedError

    def create_for_data_processing(self):
        raise NotImplementedError
