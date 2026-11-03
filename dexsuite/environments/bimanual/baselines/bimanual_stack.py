"""Bimanual stack baseline environment.

Two arms share a workspace and must stack three cubes into a tower.
"""

from __future__ import annotations

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import sample_xy_k_noncolliding
from dexsuite.utils.workspace_utils import outside_aabb

# ---------- Simulation Constants ----------
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 400

# ---------- Task Constants ----------
CUBE_SIZE = 0.05
BAND_XY = 0.25  # Area where cubes spawn
MIN_SEP = 0.15
MAX_SEP = 0.45
XY_PLACE_THR = 0.02  # Tolerance for horizontal alignment
Z_PLACE_THR = 0.01  # Tolerance for vertical gaps


@register_env("bimanual_stack")
class BimanualStackEnv(BaseEnv):
    """Bimanual cube-stacking baseline task."""

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
        self.cube_a = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.60, 0.12, 0.13)),  # Red
        )
        self.cube_b = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.29, 0.54, 0.75)),  # Blue
        )
        self.cube_c = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.18, 0.70, 0.35)),  # Green
        )

        self.upper_x_box = self.world_aabb.half("x", "high")

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        # Sample 3 non colliding positions
        a_pos, b_pos, c_pos = sample_xy_k_noncolliding(
            self.upper_x_box,
            k=3,
            band_xy=BAND_XY,
            z=CUBE_SIZE * 0.5,
            min_sep=MIN_SEP,
            max_sep=MAX_SEP,
            env_idx=envs_idx,
            mode="square",
            max_tries=100,
        )

        self.cube_a.set_pos(a_pos, envs_idx=envs_idx)
        self.cube_b.set_pos(b_pos, envs_idx=envs_idx)
        self.cube_c.set_pos(c_pos, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        return {
            "cube_a_pos": self.cube_a.get_pos(),
            "cube_b_pos": self.cube_b.get_pos(),
            "cube_c_pos": self.cube_c.get_pos(),
            "cube_a_quat": self.cube_a.get_quat(),
            "cube_b_quat": self.cube_b.get_quat(),
            "cube_c_quat": self.cube_c.get_quat(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        # It doesn't matter which arm stacked them, as long as they are stacked correctly

        cube_a_pos = obs["state"]["other"]["cube_a_pos"].reshape(-1, 3)
        cube_b_pos = obs["state"]["other"]["cube_b_pos"].reshape(-1, 3)
        cube_c_pos = obs["state"]["other"]["cube_c_pos"].reshape(-1, 3)

        is_b_centered_on_a_xy = (
            torch.norm(cube_a_pos[:, :2] - cube_b_pos[:, :2], dim=-1) < XY_PLACE_THR
        )
        # Check B is 1 cube height above A
        is_b_above_a_z = (
            torch.abs(cube_b_pos[:, 2] - (cube_a_pos[:, 2] + CUBE_SIZE)) < Z_PLACE_THR
        )

        is_c_centered_on_b_xy = (
            torch.norm(cube_b_pos[:, :2] - cube_c_pos[:, :2], dim=-1) < XY_PLACE_THR
        )
        # Check C is 1 cube height above B
        is_c_above_b_z = (
            torch.abs(cube_c_pos[:, 2] - (cube_b_pos[:, 2] + CUBE_SIZE)) < Z_PLACE_THR
        )

        return (
            is_b_centered_on_a_xy
            & is_b_above_a_z
            & is_c_centered_on_b_xy
            & is_c_above_b_z
        )

    def _is_failure(self, obs) -> torch.Tensor:
        # Failure if any cube is outside the workspace AABB
        cube_a_pos = obs["state"]["other"]["cube_a_pos"].reshape(-1, 3)
        cube_b_pos = obs["state"]["other"]["cube_b_pos"].reshape(-1, 3)
        cube_c_pos = obs["state"]["other"]["cube_c_pos"].reshape(-1, 3)

        is_cube_a_outside_aabb = outside_aabb(
            cube_a_pos,
            self.world_aabb.min,
            self.world_aabb.max,
        )
        is_cube_b_outside_aabb = outside_aabb(
            cube_b_pos,
            self.world_aabb.min,
            self.world_aabb.max,
        )
        is_cube_c_outside_aabb = outside_aabb(
            cube_c_pos,
            self.world_aabb.min,
            self.world_aabb.max,
        )

        return (
            is_cube_a_outside_aabb
            | is_cube_b_outside_aabb
            | is_cube_c_outside_aabb
            | self._is_tcp_outside_workspace(obs)
        )  # if any of the cubes are outside, or if the gripper's tcp is outside of its AABB workspace
