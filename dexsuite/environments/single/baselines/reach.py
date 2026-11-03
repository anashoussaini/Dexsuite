"""Single-arm reach baseline environment.

The robot must move its tool center point to a sampled 3D target marker.
"""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import (
    sample_in_aabb_center_xyz_band,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 100  # At 20hz this is 5 seconds

# Task
TARGET_RADIUS = 0.03
SPAWN_MARGIN = 0.05

# Reward shaping
POS_TRACK_W = -0.20  # end_effector_position_tracking
POS_FINE_W = +0.10  # end_effector_position_tracking_fine_grained
FINE_THRESH = 0.05  # meters: when the fine-grained term "hits" (ON if d < 5 cm)

ACT_RATE_W = -1.0e-3  # action_rate penalty on squared action change
JOINT_VEL_W = -1.0e-3  # joint_vel penalty on squared joint velocity

# Success
SUCCESS_THRESHOLD = 0.05  # 3 cm success, like your docstring
SUCCESS_BONUS = 10.0  # one-time on success (optional)


@register_env("reach")
class ReachEnv(BaseEnv):
    """A single-arm environment for reaching a static floating target sphere."""

    def __init__(
        self,
        *,
        robot,
        cameras,
        sim,
        render_mode: str | None = None,
        seed: int | None = None,
        **scene_kwargs,
    ):
        """Initialize the environment.

        Args:
            robot: Robot options.
            cameras: Camera options.
            sim: Simulation options.
            render_mode: Rendering mode. Use "human" to open the viewer.
            seed: Optional random seed.
            scene_kwargs: Additional keyword arguments forwarded to the underlying genesis.Scene.
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
            **scene_kwargs,
        )

    def _setup_scene(self) -> None:
        """Create the target sphere entity."""
        self.target_sphere = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=TARGET_RADIUS,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),  # Bright red
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        """Resample the target position at the start of each episode.

        Args:
            envs_idx: A tensor of environment indices to reset. If None,
                all environments are reset.
        """
        # Sample a new 3D position within a bounded box around the workspace center
        target_position = sample_in_aabb_center_xyz_band(
            self.world_aabb.with_margin(SPAWN_MARGIN),  # shrink box first
            band_xyz=(0.15, 0.20, 0.10),
            mode="ellipsoid",  # optional: uniform volume is nicer in 3D
            env_idx=envs_idx,
        )
        self.target_sphere.set_pos(target_position, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, Any]:
        """Expose the target pose in obs['state']['other'].

        Returns:
            A dictionary containing the target pose tensors.
        """
        return {
            "target_pos": self.target_sphere.get_pos(),
            "target_quat": self.target_sphere.get_quat(),
        }

    def _compute_reward(self, obs: dict[str, Any]) -> torch.Tensor:
        """Compute the reward for the reach task.

        The reward consists of:
        - Position tracking: Penalty based on distance to target
        - Fine-grained positioning: Bonus when close to target
        - Joint velocity penalty: Encourages smooth motion
        - Success bonus: One-time reward for reaching target

        Args:
            obs: The current observation dictionary.

        Returns:
            A tensor of shape (B,) containing the reward for each environment.
        """
        # Extract relevant observations
        tcp_position = obs["state"]["gripper"]["tcp_pos"]
        target_position = obs["state"]["other"]["target_pos"]
        qvel = obs["state"]["manipulator"]["qvel"]
        last_action = obs["state"]["other"]["last_action"]
        action = obs["state"]["other"]["action"]
        # Distance to target
        tcp_to_target_distance = torch.linalg.norm(
            tcp_position - target_position,
            dim=-1,
        )  # (B,)

        # Position tracking reward (negative distance penalty)
        r_pos = POS_TRACK_W * tcp_to_target_distance

        # Fine-grained positioning reward (bonus when very close)
        proximity = torch.clamp(
            (FINE_THRESH - tcp_to_target_distance) / FINE_THRESH,
            min=0.0,
            max=1.0,
        )
        r_fine = POS_FINE_W * (proximity * proximity)

        # Joint velocity penalty (encourages smooth motion)
        r_jvel = JOINT_VEL_W * torch.sum(qvel * qvel, dim=-1)

        # Action rate penalty (encourages smooth actions)
        r_arate = ACT_RATE_W * torch.sum((action - last_action) ** 2, dim=-1)

        # Success bonus (one-time reward for reaching target)
        is_success = tcp_to_target_distance < SUCCESS_THRESHOLD
        r_success = is_success.to(tcp_to_target_distance.dtype) * SUCCESS_BONUS

        # Total reward
        reward = r_pos + r_fine + r_jvel + r_success + r_arate

        return reward

    def _is_success(self, obs: dict[str, Any]) -> torch.Tensor:
        """Checks if the task is successful.

        Args:
            obs: The current observation dictionary.

        Returns:
            A boolean tensor of shape (B,) indicating success for each env.
        """
        tcp_position = obs["state"]["gripper"]["tcp_pos"]
        target_position = obs["state"]["other"]["target_pos"]

        tcp_to_target_distance = torch.norm(tcp_position - target_position, dim=-1)  # (B,)
        is_within_success_threshold = tcp_to_target_distance < SUCCESS_THRESHOLD
        return is_within_success_threshold

    # def _is_failure(self, obs) -> torch.Tensor:
    #     return self._is_tcp_outside_workspace(obs)
