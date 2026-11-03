"""Single-arm pick-and-place baseline environment.

The robot must pick a cube and place it near a fixed goal marker in the
workspace.
"""

from __future__ import annotations

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import (
    sample_in_aabb_center_xy_band,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 200  # At 20hz this is 10 seconds

# Task
CUBE_SIZE = 0.05
GOAL_R = 0.02
GOAL_Z = 0.01

# Sampling
GOAL_BAND = 0.05
CUBE_BAND = 0.05  # 5 cm radius band around the AABB center for the cube

# Success
SUCCESS_THR = 0.03


@register_env("pick_place")
class PickPlaceEnv(BaseEnv):
    """Pick a cube and place it at a goal location."""

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
        """Initialize the environment.

        Args:
            robot: Robot options.
            cameras: Camera options.
            sim: Simulation options.
            render_mode: Rendering mode. Use "human" to open the viewer.
            seed: Optional random seed.
            scene_kw: Additional keyword arguments forwarded to the underlying genesis.Scene.
        """
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
        """Create task entities and precompute sampling helpers."""
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
        )
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )

        # Top-right quadrant of the workspace (pos X & pos Y halves)
        goal_quadrant_aabb = self.world_aabb.half(axis="x", side="right").half(
            axis="y",
            side="high",
        )
        goal_quadrant_center_xyz = goal_quadrant_aabb.center()  # (3,)

        goal_center_xy = torch.stack(
            (goal_quadrant_center_xyz[0], goal_quadrant_center_xyz[1]),
            dim=0,
        ).to(self.device)
        goal_z_tensor = torch.tensor(GOAL_Z, device=self.device, dtype=torch.float32)
        self.goal_pos_fixed = torch.cat(
            (goal_center_xy, goal_z_tensor.view(1)),
            dim=0,
        )  # (3,)
        self.cube_z = float(CUBE_SIZE * 0.5)

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        """Reset cube and goal poses.

        Args:
            envs_idx: Optional subset of environments to reset (batched sim).
        """
        # Cube: sample in a 5 cm radial band around the AABB center
        cube_pos = sample_in_aabb_center_xy_band(
            self.world_aabb,
            band_xy=CUBE_BAND,
            z=self.cube_z,
            env_idx=envs_idx,
            mode="circle",
        )

        # Place cube and fixed goal
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)

        goal_quadrant_aabb = self.world_aabb.half(axis="x", side="right").half(
            axis="y",
            side="high",
        )

        goal_position_sample = sample_in_aabb_center_xy_band(
            goal_quadrant_aabb,
            band_xy=GOAL_BAND,
            z=GOAL_Z,
            env_idx=envs_idx,
            mode="circle",
        )
        self.goal_pos_fixed = goal_position_sample
        goal_pos = self.goal_pos_fixed
        self.goal.set_pos(goal_pos, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        """Expose cube and goal poses in obs['state']['other']."""
        return {
            "cube_pos": self.cube.get_pos(),
            "cube_quat": self.cube.get_quat(),
            "goal_pos": self.goal.get_pos(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Check success based on cube-to-goal distance."""
        cube_position = obs["state"]["other"]["cube_pos"]  # (3,) or (B,3)
        goal_position = obs["state"]["other"]["goal_pos"]  # (3,) or (B,3)
        distance_to_goal = torch.linalg.norm(
            cube_position - goal_position,
            dim=-1,
        )  # scalar if 1D, (B,) if 2D
        return distance_to_goal < SUCCESS_THR

    def _is_failure(self, obs) -> torch.Tensor:
        """Fail the episode if the TCP leaves the workspace."""
        return self._is_tcp_outside_workspace(obs)
