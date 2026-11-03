"""Bimanual push baseline environment.

Each arm must push its own cube across the table into a nearby goal region.
"""

from __future__ import annotations

import genesis as gs
import torch

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

# Sampling
BAND_XY = 0.10
GOAL_Y_OFFSET = -0.20

# Success / failure
SUCCESS_THR = 0.03
LIFT_THR = 0.03


@register_env("bimanual_push")
class BimanualPushEnv(BaseEnv):
    """Push two cubes to their respective goal markers."""

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
        self.left_cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
        )
        self.goal_left = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )
        self.right_cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
        )
        self.goal_right = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        cube_pos_left = sample_in_aabb_center_xy_band(
            self.world_aabb.half(axis="y", side="high"),
            band_xy=BAND_XY,
            z=CUBE_SIZE * 0.5,
            env_idx=envs_idx,
        )
        self.left_cube.set_pos(cube_pos_left, envs_idx=envs_idx)
        # left goal: shift cube along -Y, clamp to workspace
        self.goal_left.set_pos(
            self.world_aabb.clamp(
                cube_pos_left
                + torch.tensor([0.0, GOAL_Y_OFFSET, 0.0], device=self.device),
            ),
            envs_idx=envs_idx,
        )

        cube_pos_right = sample_in_aabb_center_xy_band(
            self.world_aabb.half(axis="y", side="low"),
            band_xy=BAND_XY,
            z=CUBE_SIZE * 0.5,
            env_idx=envs_idx,
        )
        self.right_cube.set_pos(cube_pos_right, envs_idx=envs_idx)
        # right goal: shift cube along +Y, clamp to workspace
        self.goal_right.set_pos(
            self.world_aabb.clamp(
                cube_pos_right
                + torch.tensor([0.0, -GOAL_Y_OFFSET, 0.0], device=self.device),
            ),
            envs_idx=envs_idx,
        )

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        return {
            "left_cube_pos": self.left_cube.get_pos(),
            "left_cube_quat": self.left_cube.get_quat(),
            "goal_left_pos": self.goal_left.get_pos(),
            "right_cube_pos": self.right_cube.get_pos(),
            "right_cube_quat": self.right_cube.get_quat(),
            "goal_right_pos": self.goal_right.get_pos(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        left_cube_position = obs["state"]["other"]["left_cube_pos"].reshape(-1, 3)
        goal_left_position = obs["state"]["other"]["goal_left_pos"].reshape(-1, 3)
        left_distance_to_goal = torch.norm(
            left_cube_position - goal_left_position,
            dim=-1,
        )

        right_cube_position = obs["state"]["other"]["right_cube_pos"].reshape(-1, 3)
        goal_right_position = obs["state"]["other"]["goal_right_pos"].reshape(-1, 3)
        right_distance_to_goal = torch.norm(
            right_cube_position - goal_right_position,
            dim=-1,
        )

        return (left_distance_to_goal < SUCCESS_THR) & (right_distance_to_goal < SUCCESS_THR)

    def _is_failure(self, obs) -> torch.Tensor:
        is_tcp_outside_workspace = self._is_tcp_outside_workspace(obs)

        left_cube_position = obs["state"]["other"]["left_cube_pos"].reshape(-1, 3)
        left_cube_z = left_cube_position[:, 2]
        is_left_cube_lifted = left_cube_z > (CUBE_SIZE * 0.5 + LIFT_THR)

        right_cube_position = obs["state"]["other"]["right_cube_pos"].reshape(-1, 3)
        right_cube_z = right_cube_position[:, 2]
        is_right_cube_lifted = right_cube_z > (CUBE_SIZE * 0.5 + LIFT_THR)

        # Failure if any cube is outside the workspace AABB

        is_left_cube_outside_aabb = outside_aabb(
            left_cube_position,
            self.world_aabb.min,
            self.world_aabb.max,
        )
        is_right_cube_outside_aabb = outside_aabb(
            right_cube_position,
            self.world_aabb.min,
            self.world_aabb.max,
        )

        return (
            is_tcp_outside_workspace
            | is_left_cube_lifted
            | is_right_cube_lifted
            | is_left_cube_outside_aabb
            | is_right_cube_outside_aabb
        )  # if either cube is lifted, or if any of the cubes are outside, or if the gripper's tcp is outside of its AABB workspace
