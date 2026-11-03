"""Single-arm stack baseline environment.

The robot must stack three cubes into a stable tower with a fixed bottom,
middle, and top ordering.
"""

from __future__ import annotations

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import sample_xy_k_noncolliding

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 600  # At 20hz this is 30 seconds

# Task
CUBE_SIZE = 0.05

# Sampling
BAND_XY = 0.2
MIN_SEP = 0.15
MAX_SEP = 0.45

# Success
XY_PLACE_THR = 0.02
Z_PLACE_THR = 0.01


@register_env("stack")
class StackEnv(BaseEnv):
    """Stack three cubes into a tower (A on bottom, then B, then C)."""

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
        """Create the three cube entities."""
        self.cube_a = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.60, 0.12, 0.13)),
        )
        self.cube_b = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.29, 0.54, 0.75)),
        )
        self.cube_c = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.18, 0.70, 0.35)),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        """Sample non-colliding start positions for the cubes.

        Args:
            envs_idx: Optional subset of environments to reset (batched sim).
        """
        a_pos, b_pos, c_pos = sample_xy_k_noncolliding(
            self.world_aabb.half("x", "high"),
            k=3,
            band_xy=BAND_XY,
            z=CUBE_SIZE * 0.5,
            min_sep=MIN_SEP,
            max_sep=MAX_SEP,
            env_idx=envs_idx,
            mode="square",
            max_tries=32,
        )
        self.cube_a.set_pos(a_pos, envs_idx=envs_idx)
        self.cube_b.set_pos(b_pos, envs_idx=envs_idx)
        self.cube_c.set_pos(c_pos, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        """Expose cube poses in obs['state']['other']."""
        return {
            "cube_a_pos": self.cube_a.get_pos(),
            "cube_b_pos": self.cube_b.get_pos(),
            "cube_c_pos": self.cube_c.get_pos(),
            "cube_a_quat": self.cube_a.get_quat(),
            "cube_b_quat": self.cube_b.get_quat(),
            "cube_c_quat": self.cube_c.get_quat(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Check whether the cubes are stacked within position tolerances."""
        cube_a_pos = obs["state"]["other"]["cube_a_pos"].reshape(-1, 3)
        cube_b_pos = obs["state"]["other"]["cube_b_pos"].reshape(-1, 3)
        cube_c_pos = obs["state"]["other"]["cube_c_pos"].reshape(-1, 3)

        is_b_centered_on_a_xy = (
            torch.norm(cube_a_pos[:, :2] - cube_b_pos[:, :2], dim=-1) < XY_PLACE_THR
        )
        is_b_above_a_z = (
            torch.abs(cube_b_pos[:, 2] - (cube_a_pos[:, 2] + CUBE_SIZE)) < Z_PLACE_THR
        ) & (cube_b_pos[:, 2] > cube_a_pos[:, 2])

        is_c_centered_on_b_xy = (
            torch.norm(cube_b_pos[:, :2] - cube_c_pos[:, :2], dim=-1) < XY_PLACE_THR
        )

        is_c_above_b_z = (
            torch.abs(cube_c_pos[:, 2] - (cube_b_pos[:, 2] + CUBE_SIZE)) < Z_PLACE_THR
        ) & (cube_c_pos[:, 2] > cube_b_pos[:, 2])

        return (
            is_b_centered_on_a_xy
            & is_b_above_a_z
            & is_c_centered_on_b_xy
            & is_c_above_b_z
        )
