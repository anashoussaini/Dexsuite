"""Bimanual pick-and-place baseline environment.

Each arm must move its own cube to a matching goal marker.
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
HORIZON = 300

# Task
CUBE_SIZE = 0.05
GOAL_R = 0.02
GOAL_Z = 0.01

# Sampling
GOAL_BAND = 0.07
CUBE_BAND = 0.10  # 5 cm radius band around the AABB center for the cube

# Success
SUCCESS_THR = 0.03


@register_env("bimanual_pick_place")
class BimanualPickPlaceEnv(BaseEnv):
    """Pick and place two cubes to their respective goal markers."""

    def __init__(
        self,
        *,
        robot,
        cameras,
        sim,
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

        goal_quadrant_left_aabb = (
            self.world_aabb.half(axis="y", side="high")
            .half(axis="x", side="right")
            .half(axis="y", side="high")
        )
        goal_quadrant_left_center_xyz = goal_quadrant_left_aabb.center()  # (3,)

        goal_left_center_xy = torch.stack(
            (goal_quadrant_left_center_xyz[0], goal_quadrant_left_center_xyz[1]),
            dim=0,
        ).to(self.device)
        goal_left_z_tensor = torch.tensor(GOAL_Z, device=self.device, dtype=torch.float32)
        self.goal_left_pos_fixed = torch.cat(
            (goal_left_center_xy, goal_left_z_tensor.view(1)),
            dim=0,
        )  # (3,)

        self.right_cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
        )
        self.goal_right = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),
        )

        goal_quadrant_right_aabb = (
            self.world_aabb.half(axis="y", side="low")
            .half(axis="x", side="right")
            .half(axis="y", side="low")
        )
        goal_quadrant_right_center_xyz = goal_quadrant_right_aabb.center()  # (3,)

        goal_right_center_xy = torch.stack(
            (goal_quadrant_right_center_xyz[0], goal_quadrant_right_center_xyz[1]),
            dim=0,
        ).to(self.device)
        goal_right_z_tensor = torch.tensor(GOAL_Z, device=self.device, dtype=torch.float32)
        self.goal_right_pos_fixed = torch.cat(
            (goal_right_center_xy, goal_right_z_tensor.view(1)),
            dim=0,
        )  # (3,)

        # Precompute fixed goal anchors for left/right halves

        self.cube_z = float(CUBE_SIZE * 0.5)

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        # Sample cubes: left on +Y side, right on -Y side of the workspace
        left_cube_pos = sample_in_aabb_center_xy_band(
            self.world_aabb.half(axis="y", side="high"),
            band_xy=CUBE_BAND,
            z=self.cube_z,
            env_idx=envs_idx,
            mode="circle",
        )
        right_cube_pos = sample_in_aabb_center_xy_band(
            self.world_aabb.half(axis="y", side="low"),
            band_xy=CUBE_BAND,
            z=self.cube_z,
            env_idx=envs_idx,
            mode="circle",
        )
        self.left_cube.set_pos(left_cube_pos, envs_idx=envs_idx)
        self.right_cube.set_pos(right_cube_pos, envs_idx=envs_idx)

        # Sample goals per side (left/right)
        goal_quadrant_left_aabb = (
            self.world_aabb.half(axis="y", side="high")
            .half(axis="x", side="right")
            .half(axis="y", side="high")
        )
        goal_left_position = sample_in_aabb_center_xy_band(
            goal_quadrant_left_aabb,
            band_xy=GOAL_BAND,
            z=GOAL_Z,
            env_idx=envs_idx,
            mode="circle",
        )
        goal_quadrant_right_aabb = (
            self.world_aabb.half(axis="y", side="low")
            .half(axis="x", side="right")
            .half(axis="y", side="low")
        )
        goal_right_position = sample_in_aabb_center_xy_band(
            goal_quadrant_right_aabb,
            band_xy=GOAL_BAND,
            z=GOAL_Z,
            env_idx=envs_idx,
            mode="circle",
        )

        self.goal_left.set_pos(goal_left_position, envs_idx=envs_idx)
        self.goal_right.set_pos(goal_right_position, envs_idx=envs_idx)

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
        left_distance_to_goal = torch.norm(left_cube_position - goal_left_position, dim=-1)

        right_cube_position = obs["state"]["other"]["right_cube_pos"].reshape(-1, 3)
        goal_right_position = obs["state"]["other"]["goal_right_pos"].reshape(-1, 3)
        right_distance_to_goal = torch.norm(
            right_cube_position - goal_right_position,
            dim=-1,
        )

        return (left_distance_to_goal < SUCCESS_THR) & (right_distance_to_goal < SUCCESS_THR)

    def _is_failure(self, obs) -> torch.Tensor:
        left_cube_pos = obs["state"]["other"]["left_cube_pos"].reshape(-1, 3)
        right_cube_pos = obs["state"]["other"]["right_cube_pos"].reshape(-1, 3)

        is_left_cube_outside_aabb = outside_aabb(
            left_cube_pos,
            self.world_aabb.min,
            self.world_aabb.max,
        )
        is_right_cube_outside_aabb = outside_aabb(
            right_cube_pos, self.world_aabb.min, self.world_aabb.max,
        )

        return (
            self._is_tcp_outside_workspace(obs)
            | is_left_cube_outside_aabb
            | is_right_cube_outside_aabb
        )
