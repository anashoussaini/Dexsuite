"""Single-arm push baseline environment.

The robot must push a cube across the table into a goal region without lifting
it off the surface.
"""

from __future__ import annotations

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import sample_in_aabb_center_xy_band

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 200  # At 20hz this is 10 seconds

# Task
CUBE_SIZE = 0.05
GOAL_R = 0.02

# Sampling
BAND_XY = 0.10
GOAL_Y_OFFSET = -0.20

# Success / failure
SUCCESS_THR = 0.03
LIFT_THR = 0.03


@register_env("push")
class PushEnv(BaseEnv):
    """Push a cube to a nearby goal marker."""

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
        """Create task entities (cube + goal marker)."""
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(),
        )
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        """Reset cube and goal poses.

        The goal is placed at a fixed offset from the cube spawn and then
        clamped to the workspace AABB.

        Args:
            envs_idx: Optional subset of environments to reset (batched sim).
        """
        cube_pos = sample_in_aabb_center_xy_band(
            self.world_aabb,
            band_xy=BAND_XY,
            z=CUBE_SIZE * 0.5,
            env_idx=envs_idx,
        )
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)

        # Legacy quirk: Allegro's teleop/config pipeline is mirrored relative
        # to Robotiq/UMI for this task, so flip the goal offset sign only for
        # that embodiment.
        goal_y_offset = GOAL_Y_OFFSET
        if getattr(self.robot_options.single, "gripper", None) == "allegro":
            goal_y_offset = -GOAL_Y_OFFSET

        # Goal is cube shifted -0.20 m in Y, clamped to the workspace.
        self.goal.set_pos(
            self.world_aabb.clamp(
                cube_pos + torch.tensor([0.0, goal_y_offset, 0.0], device=self.device),
            ),
            envs_idx=envs_idx,
        )

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        """Expose cube and goal poses in obs['state']['other']."""
        return {
            "cube_pos": self.cube.get_pos(),
            "cube_quat": self.cube.get_quat(),
            "goal_pos": self.goal.get_pos(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Check success based on cube-to-goal distance."""
        cube_position = obs["state"]["other"]["cube_pos"].reshape(-1, 3)
        goal_position = obs["state"]["other"]["goal_pos"].reshape(-1, 3)
        distance_to_goal = torch.norm(cube_position - goal_position, dim=-1)
        return distance_to_goal < SUCCESS_THR

    def _is_failure(self, obs) -> torch.Tensor:
        """Fail if TCP leaves workspace or cube is lifted off the table."""
        is_tcp_outside_workspace = self._is_tcp_outside_workspace(obs)
        cube_position = obs["state"]["other"]["cube_pos"].reshape(-1, 3)
        cube_z = cube_position[:, 2]
        is_cube_lifted = cube_z > (CUBE_SIZE * 0.5 + LIFT_THR)
        return is_tcp_outside_workspace | is_cube_lifted
