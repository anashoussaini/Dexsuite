"""Bimanual lift baseline environment.

Each arm must lift its own cube to a goal marker above the spawn position.
"""

from __future__ import annotations

import genesis as gs
import torch
from torch import Tensor

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import sample_in_aabb_center_xy_band
from dexsuite.utils.workspace_utils import outside_aabb

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 200

# Task
CUBE_SIZE = 0.05
GOAL_R = 0.02
GOAL_UP = 0.20 + CUBE_SIZE / 2.0
LIFT_EPS = 0.01

# Success
SUCCESS_XY_THRES = 0.05  # XY distance threshold to consider the cube aligned

# Reward shaping (kept for tuning; not all are used directly)
CENTER_BAND = 0.25
TCP_THR = 0.05
STD = 0.3

POS_TRACK_W = -0.20  # end_effector_position_tracking
POS_FINE_W = +0.10  # end_effector_position_tracking_fine_grained
GRSP_W = 0.5  # grasp reward weight
LIFT_W = 1.0  # lift reward weight
FINE_THRESH = 0.05  # meters: when the fine-grained term "hits" (ON if d < 5 cm)
ACT_RATE_W = -1.0e-4  # action_rate penalty on ||a - a_prev||^2
JOINT_VEL_W = -1.0e-4  # joint_vel   penalty on ||qdot||^2
SUCCESS_THRESHOLD = 0.03  # 3 cm success, like your docstring
SUCCESS_BONUS = 1.0  # one-time on success (optional)
TIME_PENALTY = 1.0e-3  # small per-step penalty


@register_env("bimanual_lift")
class BimanualLiftEnv(BaseEnv):
    """Lift two cubes (left/right) to a fixed height above their spawns."""

    def __init__(
        self,
        *,
        robot=None,
        cameras=None,
        sim=None,
        render_mode: str | None = None,
        seed: int | None = None,
        **scene_kw,
    ):
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
        self.right_cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.10, 0.24, 0.84)),
        )
        self.goal_right = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),
        )

        self.left_cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.90, 0.76, 0.16)),
        )
        self.goal_left = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )

    def _on_episode_start(self, envs_idx: Tensor | None) -> None:
        # cubes start on the table; each side lifts its own cube straight up
        cube_pos_right = sample_in_aabb_center_xy_band(
            self.world_aabb.half("y", "low"),
            band_xy=(0.1, 0.25),
            z=CUBE_SIZE / 2.0,
            env_idx=envs_idx,
            mode="square",
        )
        self.right_cube.set_pos(cube_pos_right, envs_idx=envs_idx)
        goal_pos_right = cube_pos_right + torch.as_tensor(
            [0.0, 0.0, GOAL_UP],
            device=cube_pos_right.device,
            dtype=cube_pos_right.dtype,
        )
        self.goal_right.set_pos(goal_pos_right, envs_idx=envs_idx)

        cube_pos_left = sample_in_aabb_center_xy_band(
            self.world_aabb.half("y", "high"),
            band_xy=(0.1, 0.25),
            z=CUBE_SIZE / 2.0,
            env_idx=envs_idx,
            mode="square",
        )
        self.left_cube.set_pos(cube_pos_left, envs_idx=envs_idx)
        goal_pos_left = cube_pos_left + torch.as_tensor(
            [0.0, 0.0, GOAL_UP],
            device=cube_pos_left.device,
            dtype=cube_pos_left.dtype,
        )
        self.goal_left.set_pos(goal_pos_left, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, Tensor]:
        return {
            "right_cube_pos": self.right_cube.get_pos(),
            "right_cube_quat": self.right_cube.get_quat(),
            "target_right_pos": self.goal_right.get_pos(),
            "left_cube_pos": self.left_cube.get_pos(),
            "left_cube_quat": self.left_cube.get_quat(),
            "target_left_pos": self.goal_left.get_pos(),
        }

    def _is_success(self, obs) -> Tensor:
        left_cube_position = obs["state"]["other"]["left_cube_pos"]
        left_target_position = obs["state"]["other"]["target_left_pos"]
        right_cube_position = obs["state"]["other"]["right_cube_pos"]
        right_target_position = obs["state"]["other"]["target_right_pos"]
        # Success if cube height reaches the target height (with tolerance).
        target_z = GOAL_UP - LIFT_EPS

        is_left_height_reached = left_cube_position[..., 2] >= target_z
        xy_dist_left = torch.linalg.norm(
            left_cube_position[..., :2] - left_target_position[..., :2],
            dim=-1,
        )
        # success if xy distance is below threshold
        is_left_xy_aligned = xy_dist_left <= SUCCESS_XY_THRES

        is_right_height_reached = right_cube_position[..., 2] >= target_z
        xy_dist_right = torch.linalg.norm(
            right_cube_position[..., :2] - right_target_position[..., :2],
            dim=-1,
        )
        # success if xy distance is below threshold
        is_right_xy_aligned = xy_dist_right <= SUCCESS_XY_THRES

        return (
            is_left_xy_aligned
            & is_left_height_reached
            & is_right_xy_aligned
            & is_right_height_reached
        )

    def _is_failure(self, obs) -> Tensor:
        left_cube_pos = obs["state"]["other"]["left_cube_pos"].reshape(-1, 3)
        right_cube_pos = obs["state"]["other"]["right_cube_pos"].reshape(-1, 3)
        left_out = outside_aabb(left_cube_pos, self.world_aabb.min, self.world_aabb.max)
        right_out = outside_aabb(
            right_cube_pos, self.world_aabb.min, self.world_aabb.max,
        )

        return self._is_tcp_outside_workspace(obs) | left_out | right_out
