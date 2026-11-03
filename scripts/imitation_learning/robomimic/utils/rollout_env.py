"""Patch DexSuite robomimic rollouts for policy-compatible observations."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

_VISUAL_KEY_MARKERS = ("_rgb", "_depth", "_image", "camera", "cameras")
_QUAT_EPS = 1e-6


def obs_keys_from_config(config: Any) -> tuple[str, ...]:
    """Return policy observation keys declared in a robomimic config."""
    obs_modalities = config.observation.modalities.obs
    keys: list[str] = []
    for modality in ("low_dim", "rgb", "depth", "scan"):
        keys.extend(getattr(obs_modalities, modality, []) or [])
    return tuple(dict.fromkeys(str(key) for key in keys))


def obs_keys_need_visual_obs(obs_keys: Iterable[str] | None) -> bool:
    """Return whether policy observations require rendered camera tensors."""
    return any(
        any(marker in str(key).lower() for marker in _VISUAL_KEY_MARKERS)
        for key in obs_keys or ()
    )


def ensure_dexsuite_rollout_patches(
    obs_keys: Iterable[str] | None = None,
) -> None:
    """Patch DexSuite robomimic envs for action safety and obs-key filtering."""
    env_dexsuite = _import_env_dexsuite()

    from dexsuite.environments import base_env

    env_path = getattr(env_dexsuite, "__file__", "<unknown>")
    print(f"[robomimic] using EnvDexsuite from: {env_path}")

    env_class = env_dexsuite.EnvDexsuite
    env_class._dexsuite_policy_obs_keys = (
        frozenset(str(key) for key in obs_keys) if obs_keys else None
    )
    env_class._dexsuite_policy_needs_visual_obs = obs_keys_need_visual_obs(obs_keys)

    if getattr(env_class, "_dexsuite_rollout_patched", False):
        return

    _install_base_env_patch(base_env.BaseEnv)
    _install_envdexsuite_patch(env_class)
    env_class._dexsuite_rollout_patched = True
    print("[robomimic] patched EnvDexsuite rollout action, observation, and video handling")


def _import_env_dexsuite():
    """Return the EnvDexsuite module, registering the local copy with robomimic."""
    import sys

    try:
        from robomimic.envs import env_dexsuite
    except ImportError:
        from . import env_dexsuite

        sys.modules["robomimic.envs.env_dexsuite"] = env_dexsuite
    _install_env_registry_patch(env_dexsuite)
    return env_dexsuite


def _install_env_registry_patch(env_dexsuite_module) -> None:
    """Teach robomimic's env resolution about EnvType.DEXSUITE_TYPE (4)."""
    import robomimic.envs.env_base as EB
    import robomimic.utils.env_utils as EnvUtils

    if not hasattr(EB.EnvType, "DEXSUITE_TYPE"):
        EB.EnvType.DEXSUITE_TYPE = 4
    if getattr(EnvUtils.get_env_class, "_dexsuite_registered", False):
        return

    original_get_env_class = EnvUtils.get_env_class

    def get_env_class(env_meta=None, env_type=None, env=None):
        resolved = EnvUtils.get_env_type(env_meta=env_meta, env_type=env_type, env=env)
        if resolved == EB.EnvType.DEXSUITE_TYPE:
            return env_dexsuite_module.EnvDexsuite
        return original_get_env_class(env_type=resolved)

    get_env_class._dexsuite_registered = True
    EnvUtils.get_env_class = get_env_class


def _install_base_env_patch(base_env_class) -> None:
    base_env_class._dexsuite_original_make_obs = base_env_class._make_obs
    base_env_class._make_obs = _make_obs_without_unneeded_camera_obs


def _install_envdexsuite_patch(env_class) -> None:
    env_class._dexsuite_original_init = env_class.__init__
    env_class._dexsuite_original_reset = env_class.reset
    env_class._dexsuite_original_step = env_class.step
    env_class._dexsuite_original_get_observation = env_class.get_observation
    env_class._dexsuite_original_render = env_class.render
    env_class.__init__ = _envdexsuite_init
    env_class._find_genesis_abs_quat_slices = _find_genesis_abs_quat_slices
    env_class._sanitize_action = _sanitize_action
    env_class._filter_policy_obs = _filter_policy_obs
    env_class._render_cached_or_live_camera = _render_cached_or_live_camera
    env_class.reset = _envdexsuite_reset
    env_class.step = _envdexsuite_step
    env_class.get_observation = _envdexsuite_get_observation
    env_class.render = _envdexsuite_render


def _envdexsuite_init(
    self,
    env_name,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
    use_depth_obs=False,
    **env_config,
):
    needs_visual_obs = bool(
        getattr(self.__class__, "_dexsuite_policy_needs_visual_obs", False),
    )
    init_config = _envdexsuite_init_config(
        env_config,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        needs_visual_obs=needs_visual_obs,
    )
    self._dexsuite_original_init(
        env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        **init_config,
    )
    self._dexsuite_policy_needs_visual_obs = needs_visual_obs
    self._env._dexsuite_skip_camera_obs = not needs_visual_obs


def _envdexsuite_init_config(
    env_config: dict[str, Any],
    *,
    render: bool,
    render_offscreen: bool,
    use_image_obs: bool,
    use_depth_obs: bool,
    needs_visual_obs: bool,
) -> dict[str, Any]:
    needs_any_camera = (
        render or render_offscreen or use_image_obs or use_depth_obs or needs_visual_obs
    )
    return env_config if needs_any_camera else _disable_env_cameras(env_config)


def _make_obs_without_unneeded_camera_obs(self):
    if not getattr(self, "_dexsuite_skip_camera_obs", False):
        return self._dexsuite_original_make_obs()
    return {"state": _state_obs_with_action_terms(self), "cameras": {}}


def _state_obs_with_action_terms(env) -> dict[str, Any]:
    import torch

    state = env.robot.get_obs()
    extras = env._get_extra_obs() or {}
    for key, value in extras.items():
        tensor = (
            value if torch.is_tensor(value) else torch.as_tensor(value, device=env.device)
        )
        state.setdefault("other", {})[key] = tensor
    state.setdefault("other", {})["action"] = env._current_action.clone()
    state.setdefault("other", {})["last_action"] = env._last_action.clone()
    return state


def _envdexsuite_reset(self):
    return self._filter_policy_obs(self._dexsuite_original_reset())


def _envdexsuite_step(self, action):
    obs, reward, done, info = self._dexsuite_original_step(self._sanitize_action(action))
    return self._filter_policy_obs(obs), reward, done, info


def _envdexsuite_get_observation(self, obs=None):
    return self._filter_policy_obs(self._dexsuite_original_get_observation(obs))


def _envdexsuite_render(self, mode="human", height=None, width=None, camera_name=None):
    if mode == "rgb_array":
        return self._render_cached_or_live_camera(camera_name, height, width)
    return self._dexsuite_original_render(
        mode=mode,
        height=height,
        width=width,
        camera_name=camera_name,
    )


def _find_genesis_abs_quat_slices(self):
    layout = getattr(getattr(self._env, "robot", None), "_layout", None)
    if layout is None:
        return []

    slices = []
    for seg in getattr(layout, "segments", ()):
        ctrl = getattr(seg, "ctrl", None)
        if _is_genesis_abs_quat_segment(ctrl, seg):
            slices.append((int(seg.start), int(seg.stop)))
    return slices


def _is_genesis_abs_quat_segment(ctrl, seg) -> bool:
    return (
        ctrl is not None
        and ctrl.__class__.__name__ == "OSCPoseAbsoluteQuatController"
        and getattr(ctrl, "backend", None) == "genesis_internal"
        and int(seg.stop) - int(seg.start) >= 7
    )


def _sanitize_action(self, action):
    import numpy as np
    import torch

    act_t = _action_to_tensor(self, action, np=np, torch=torch)
    if act_t.shape[-1] != self._action_dim:
        raise ValueError(
            f"Expected action dim {self._action_dim}, got shape {tuple(act_t.shape)}",
        )

    _ensure_action_bounds(self, torch=torch)
    _ensure_abs_quat_slices(self)
    act_t = torch.nan_to_num(act_t, nan=0.0, posinf=0.0, neginf=0.0)
    act_t = torch.minimum(torch.maximum(act_t, self._action_low), self._action_high)
    _normalize_abs_quat_actions(act_t, self._genesis_abs_quat_slices, torch=torch)
    return act_t


def _action_to_tensor(env, action, *, np, torch):
    if isinstance(action, np.ndarray):
        return torch.from_numpy(action).to(env._device, dtype=torch.float32).clone()
    if torch.is_tensor(action):
        return action.to(env._device, dtype=torch.float32).clone()
    raise TypeError(f"action must be np.ndarray or torch.Tensor, got {type(action)}")


def _ensure_action_bounds(env, *, torch) -> None:
    if hasattr(env, "_action_low"):
        return
    env._action_low = torch.as_tensor(
        env._env.action_space.low,
        device=env._device,
        dtype=torch.float32,
    )
    env._action_high = torch.as_tensor(
        env._env.action_space.high,
        device=env._device,
        dtype=torch.float32,
    )


def _ensure_abs_quat_slices(env) -> None:
    if not hasattr(env, "_genesis_abs_quat_slices"):
        env._genesis_abs_quat_slices = env._find_genesis_abs_quat_slices()


def _normalize_abs_quat_actions(action, slices, *, torch) -> None:
    action_2d = action.unsqueeze(0) if action.ndim == 1 else action
    for start, _stop in slices:
        quat = action_2d[..., start + 3 : start + 7]
        quat_norm = quat.norm(dim=-1, keepdim=True)
        quat_identity = torch.zeros_like(quat)
        quat_identity[..., 0] = 1.0
        quat.copy_(
            torch.where(
                quat_norm > _QUAT_EPS,
                quat / quat_norm.clamp_min(_QUAT_EPS),
                quat_identity,
            ),
        )


def _filter_policy_obs(self, obs):
    keys = getattr(self.__class__, "_dexsuite_policy_obs_keys", None)
    if not keys or not isinstance(obs, dict):
        return obs
    return {key: value for key, value in obs.items() if key in keys}


def _render_cached_or_live_camera(self, camera_name, height, width):
    frame = _cached_camera_frame(self, camera_name)
    if frame is None:
        frame = _render_live_camera_frame(self, camera_name)
    return _format_video_frame(frame, height=height, width=width)


def _disable_env_cameras(env_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(env_config)
    options = config.get("options", config)
    options["cameras"] = {"static": {}, "dynamic": {}, "modalities": ["rgb"]}
    return config


def _cached_camera_frame(env, camera_name: str | None):
    obs = getattr(env, "_last_obs_np", None)
    if not isinstance(obs, dict):
        return None

    names = [camera_name, "front"]
    for name in names:
        if name:
            frame = obs.get(f"cameras_{name}_rgb")
            if frame is not None:
                return frame

    for key, frame in obs.items():
        if str(key).startswith("cameras_") and str(key).endswith("_rgb"):
            return frame
    return None


def _render_live_camera_frame(env, camera_name: str | None):
    camera_sys = getattr(getattr(env, "_env", None), "camera_sys", None)
    cameras = getattr(camera_sys, "_cams", {})
    if not cameras:
        return None

    selected = camera_name if camera_name in cameras else None
    if selected is None and "front" in cameras:
        selected = "front"
    if selected is None:
        selected = next(iter(cameras))

    camera, _spec = cameras[selected]
    frames = camera.render(depth=False, segmentation=False, normal=False)
    return frames[0] if frames else None


def _format_video_frame(frame, *, height: int | None, width: int | None):
    import numpy as np

    out_height = int(height or 512)
    out_width = int(width or 512)
    if frame is None:
        return np.zeros((out_height, out_width, 3), dtype=np.uint8)

    arr = _frame_to_hwc_array(frame)
    if arr is None:
        return np.zeros((out_height, out_width, 3), dtype=np.uint8)

    arr = _frame_to_uint8(arr)
    if (arr.shape[0], arr.shape[1]) != (out_height, out_width):
        arr = _resize_frame_nearest(arr, out_height, out_width)
    return arr


def _frame_to_hwc_array(frame):
    import numpy as np
    import torch

    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3:
        return None
    return arr[..., :3]


def _frame_to_uint8(arr):
    import numpy as np

    if arr.dtype != np.uint8:
        if arr.size and arr.max() <= 1.0 and arr.min() >= 0.0:
            return (arr * 255.0).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_frame_nearest(arr, height: int, width: int):
    import numpy as np

    yy = np.linspace(0, arr.shape[0] - 1, height).astype(np.int64)
    xx = np.linspace(0, arr.shape[1] - 1, width).astype(np.int64)
    return arr[yy][:, xx]
