"""Single-arm tool-hang rigid environment.

The robot must hang a ratcheting wrench on a hook frame mounted to a stand.
"""

from __future__ import annotations

import math
from typing import Any

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.orientation_utils import (
    quat_to_rotmat_wxyz_torch,
    rpy_to_quat_wxyz_torch,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 1000

# Robosuite-inspired placement (offsets from stand in world XY).
# Robosuite (table frame): stand x=-0.08, frame x=-0.04, tool x=0.04  -> relative: +0.04, +0.12
FRAME_X_CENTER = 0.04
FRAME_Y_CENTER = -0.24
TOOL_X_CENTER = 0.12
TOOL_Y_CENTER = -0.20

FRAME_XY_TOL = 0.02
TOOL_XY_TOL = 0.02

# Robosuite-inspired default rotations.
FRAME_PITCH_CENTER = (-math.pi / 2.0) + (math.pi / 6.0)
FRAME_PITCH_TOL = math.pi / 18.0
TOOL_YAW_CENTER = (-math.pi / 2.0) - (math.pi / 9.0)
TOOL_YAW_TOL = math.pi / 18.0

# Success checks (geometry only; no contact checks).
BASE_UP_COS_THR = math.cos(math.pi / 18.0)  # 10 degrees
FRAME_INSERT_DIST_THR = 0.05
TOOL_LINE_DIST_THR = 0.0105 - 0.00375  # inner_radius_1 - frame_thickness/2
TOOL_INSERT_MIN = 0.05

# Local site/feature points (from the MJCF assets, body frame coordinates).
STAND_BASE_CENTER_LOCAL = (0.0, 0.0, -0.075)
FRAME_HANG_SITE_LOCAL = (-0.0475, 0.0, 0.08625)
FRAME_INTERSECTION_SITE_LOCAL = (0.04375, 0.0, 0.08625)
FRAME_TIP_SITE_LOCAL = (0.04375, 0.0, -0.13445)
TOOL_HOLE1_CENTER_LOCAL = (-0.093, 0.0, 0.0)


@register_env("tool_hang")
class ToolHangEnv(BaseEnv):
    """Rigid single-arm environment for hanging a wrench on a hook frame."""

    def _batch_size(self, envs_idx: torch.Tensor | None) -> int:
        return int(self.n_envs) if envs_idx is None else int(envs_idx.numel())

    def _maybe_squeeze0(self, x: torch.Tensor, envs_idx: torch.Tensor | None) -> torch.Tensor:
        if envs_idx is None and self.n_envs == 1 and x.ndim == 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x

    @staticmethod
    def _site_world_pos(
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        site_local_xyz: tuple[float, float, float],
    ) -> torch.Tensor:
        local = torch.as_tensor(site_local_xyz, dtype=body_pos.dtype, device=body_pos.device)
        return body_pos + (quat_to_rotmat_wxyz_torch(body_quat) @ local)

    def __init__(
        self,
        *,
        robot,
        cameras,
        sim,
        render_mode: str | None,
        seed: int | None = None,
        **scene_kw,
    ):
        """Initialize the tool-hang environment."""
        super().__init__(
            robot_options=robot,
            cameras_options=cameras,
            sim_options=sim,
            render_mode=render_mode,
            seed=seed,
            sim_dt=SIM_DT,
            substeps=SUBSTEPS,
            horizon=HORIZON,
            **scene_kw,
        )

    def _setup_scene(self) -> None:
        wrench_path = get_object_path("task/ratcheting_wrench")
        hook_frame_path = get_object_path("task/hook_frame")
        stand_path = get_object_path("task/stand_with_mount")

        self.wrench = self.scene.add_entity(
            gs.morphs.MJCF(file=str(wrench_path), collision=True),
            material=gs.materials.Rigid(),
        )
        self.hook_frame = self.scene.add_entity(
            gs.morphs.MJCF(file=str(hook_frame_path), collision=True),
            material=gs.materials.Rigid(),
        )
        self.stand_with_mount = self.scene.add_entity(
            gs.morphs.MJCF(file=str(stand_path), collision=True),
            material=gs.materials.Rigid(),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        batch_size = self._batch_size(envs_idx)
        dev = self.device

        stand_center = self.world_aabb.center().to(device=dev, dtype=torch.float32).reshape(1, 3)
        stand_pos = stand_center.repeat(batch_size, 1)
        self.stand_with_mount.set_pos(self._maybe_squeeze0(stand_pos, envs_idx), envs_idx=envs_idx)

        stand_xy = stand_center[:, :2]
        frame_xy_anchor = stand_xy + torch.tensor(
            [FRAME_X_CENTER, FRAME_Y_CENTER],
            dtype=torch.float32,
            device=dev,
        ).reshape(1, 2)
        tool_xy_anchor = stand_xy + torch.tensor(
            [TOOL_X_CENTER, TOOL_Y_CENTER],
            dtype=torch.float32,
            device=dev,
        ).reshape(1, 2)

        frame_xy = frame_xy_anchor + (torch.rand(batch_size, 2, device=dev) * 2.0 - 1.0) * FRAME_XY_TOL
        tool_xy = tool_xy_anchor + (torch.rand(batch_size, 2, device=dev) * 2.0 - 1.0) * TOOL_XY_TOL

        frame_pos = torch.cat([frame_xy, stand_pos[:, 2:3]], dim=1)
        tool_pos = torch.cat([tool_xy, stand_pos[:, 2:3]], dim=1)

        self.hook_frame.set_pos(self._maybe_squeeze0(frame_pos, envs_idx), envs_idx=envs_idx)
        self.wrench.set_pos(self._maybe_squeeze0(tool_pos, envs_idx), envs_idx=envs_idx)

        stand_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev).reshape(1, 4)
        self.stand_with_mount.set_quat(
        self._maybe_squeeze0(stand_quat.repeat(batch_size, 1), envs_idx),
        envs_idx=envs_idx,
        )

        frame_pitch = FRAME_PITCH_CENTER + (torch.rand(batch_size, device=dev) * 2.0 - 1.0) * FRAME_PITCH_TOL
        frame_quat = rpy_to_quat_wxyz_torch(
            torch.stack(
                [
                    torch.zeros(batch_size, device=dev),
                    frame_pitch,
                    torch.zeros(batch_size, device=dev),
                ],
                dim=1,
            ),
        )
        self.hook_frame.set_quat(self._maybe_squeeze0(frame_quat, envs_idx), envs_idx=envs_idx)

        tool_yaw = TOOL_YAW_CENTER + (torch.rand(batch_size, device=dev) * 2.0 - 1.0) * TOOL_YAW_TOL
        tool_quat = rpy_to_quat_wxyz_torch(
            torch.stack(
                [
                    torch.zeros(batch_size, device=dev),
                    torch.zeros(batch_size, device=dev),
                    tool_yaw,
                ],
                dim=1,
            ),
        )
        self.wrench.set_quat(self._maybe_squeeze0(tool_quat, envs_idx), envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, Any]:
        return {
            "wrench_pos": self.wrench.get_pos(),
            "wrench_quat": self.wrench.get_quat(),
            "hook_frame_pos": self.hook_frame.get_pos(),
            "hook_frame_quat": self.hook_frame.get_quat(),
            "stand_pos": self.stand_with_mount.get_pos(),
            "stand_quat": self.stand_with_mount.get_quat(),
        }

    def _check_frame_assembled(self, other: dict[str, torch.Tensor]) -> torch.Tensor:
        stand_pos = other["stand_pos"]
        stand_quat = other["stand_quat"]
        frame_pos = other["hook_frame_pos"]
        frame_quat = other["hook_frame_quat"]

        stand_up = quat_to_rotmat_wxyz_torch(stand_quat)[..., :, 2]
        base_shaft_is_vertical = stand_up[..., 2] >= BASE_UP_COS_THR

        base_pos = self._site_world_pos(stand_pos, stand_quat, STAND_BASE_CENTER_LOCAL)
        bottom_hook_pos = self._site_world_pos(frame_pos, frame_quat, FRAME_TIP_SITE_LOCAL)
        insertion_dist = torch.linalg.norm(bottom_hook_pos - base_pos, dim=-1)
        bottom_is_close_enough = insertion_dist < FRAME_INSERT_DIST_THR

        return (base_shaft_is_vertical & bottom_is_close_enough).reshape(-1)

    def _check_tool_on_frame(self, other: dict[str, torch.Tensor]) -> torch.Tensor:
        tool_pos = other["wrench_pos"]
        tool_quat = other["wrench_quat"]
        frame_pos = other["hook_frame_pos"]
        frame_quat = other["hook_frame_quat"]

        hook_endpoint = self._site_world_pos(frame_pos, frame_quat, FRAME_HANG_SITE_LOCAL)
        hook_corner = self._site_world_pos(frame_pos, frame_quat, FRAME_INTERSECTION_SITE_LOCAL)
        frame_hook_vec = hook_corner - hook_endpoint
        frame_hook_len = torch.linalg.norm(frame_hook_vec, dim=-1).clamp_min(1e-9)
        frame_hook_dir = frame_hook_vec / frame_hook_len.unsqueeze(-1)

        tool_hole_center = self._site_world_pos(tool_pos, tool_quat, TOOL_HOLE1_CENTER_LOCAL)
        tool_vec = tool_hole_center - hook_endpoint
        tool_dot = (tool_vec * frame_hook_dir).sum(dim=-1)
        tool_proj = tool_dot.unsqueeze(-1) * frame_hook_dir
        dist_to_hook_line = torch.linalg.norm(tool_vec - tool_proj, dim=-1)
        tool_hole_is_close_enough = dist_to_hook_line < TOOL_LINE_DIST_THR

        normalized_dist = tool_dot / frame_hook_len
        tool_is_inserted_far_enough = (normalized_dist > TOOL_INSERT_MIN) & (normalized_dist < 1.0)

        return (tool_hole_is_close_enough & tool_is_inserted_far_enough).reshape(-1)

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        return self._check_frame_assembled(other) & self._check_tool_on_frame(other)

    def _compute_reward(self, obs) -> torch.Tensor:
        return self._is_success(obs).to(dtype=torch.float32, device=self.device)
