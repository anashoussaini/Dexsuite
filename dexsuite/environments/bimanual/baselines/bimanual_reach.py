"""Bimanual reach baseline environment.

Each arm must move its tool center point to an independent sampled target
marker.
"""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch
from torch import Tensor

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.randomizers import (
    sample_in_aabb_center_xyz_band,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 200

# Task
TARGET_RADIUS = 0.03
SPAWN_MARGIN = 0.02

# Reward shaping (kept for tuning; not all are used directly)
POS_TRACK_W = -0.20  # end_effector_position_tracking
POS_FINE_W = +0.10  # end_effector_position_tracking_fine_grained
FINE_THRESH = 0.05  # meters: when the fine-grained term "hits" (ON if d < 5 cm)

ACT_RATE_W = -1.0e-3  # action_rate penalty on squared action change
JOINT_VEL_W = -1.0e-3  # joint_vel penalty on squared joint velocity

# Success
SUCCESS_THRESHOLD = 0.03  # 3 cm success


@register_env("bimanual_reach")
class BimanualReachEnv(BaseEnv):
    """A bimanual-arm environment for reaching a static floating target sphere."""

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
        """Adds left/right target spheres to the simulation scene."""
        # Call base to set up robot, ground plane, etc.
        self.target_sphere_left = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=TARGET_RADIUS,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),  # Bright red
        )
        self.target_sphere_right = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=TARGET_RADIUS,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),  # Bright red
        )

    def _on_episode_start(self, envs_idx: Tensor | None) -> None:
        """Resamples left/right target positions at the start of each episode.

        Args:
            envs_idx: A tensor of environment indices to reset. If None,
                all environments are reset.
        """
        # Sample new positions within bounded boxes on each half of the workspace
        target_position_left = sample_in_aabb_center_xyz_band(
            self.world_aabb.half("y", "high").with_margin(
                SPAWN_MARGIN,
            ),  # shrink box first
            band_xyz=(0.17, 0.27, 0.15),
            mode="ellipsoid",  # optional: uniform volume is nicer in 3D
            env_idx=envs_idx,
        )
        self.target_sphere_left.set_pos(target_position_left, envs_idx=envs_idx)

        target_position_right = sample_in_aabb_center_xyz_band(
            self.world_aabb.half("y", "low").with_margin(
                SPAWN_MARGIN,
            ),  # shrink box first
            band_xyz=(0.17, 0.27, 0.15),
            mode="ellipsoid",  # optional: uniform volume is nicer in 3D
            env_idx=envs_idx,
        )
        self.target_sphere_right.set_pos(target_position_right, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, Any]:
        """Appends the target's position to the observation dictionary.

        Returns:
            A dictionary containing the target's 3D position.
        """
        return {
            "target_left_pos": self.target_sphere_left.get_pos(),
            "target_left_quat": self.target_sphere_left.get_quat(),
            "target_right_pos": self.target_sphere_right.get_pos(),
            "target_right_quat": self.target_sphere_right.get_quat(),
        }

    # def _compute_reward(self, obs: dict[str, Any]) -> Tensor:
    #     """Compute the reward for the reach task.

    #     The reward consists of:
    #     - Position tracking: Penalty based on distance to target
    #     - Fine-grained positioning: Bonus when close to target
    #     - Joint velocity penalty: Encourages smooth motion
    #     - Success bonus: One-time reward for reaching target

    #     Args:
    #         obs: The current observation dictionary.

    #     Returns:
    #         A tensor of shape (B,) containing the reward for each environment.
    #     """
    #     # Extract relevant observations
    #     tcp_position = obs["state"]["gripper"]["tcp_pos"]
    #     target_position = obs["state"]["other"]["target_pos"]
    #     qvel = obs["state"]["manipulator"]["qvel"]
    #     last_action = obs["state"]["other"]["last_action"]
    #     action = obs["state"]["other"]["action"]
    #     # Distance to target
    #     distance = torch.linalg.norm(tcp_position - target_position, dim=-1)  # (B,)

    #     # Position tracking reward (negative distance penalty)
    #     r_pos = POS_TRACK_W * distance

    #     # Fine-grained positioning reward (bonus when very close)
    #     proximity = torch.clamp(
    #         (FINE_THRESH - distance) / FINE_THRESH, min=0.0, max=1.0
    #     )
    #     r_fine = POS_FINE_W * (proximity * proximity)

    #     # Joint velocity penalty (encourages smooth motion)
    #     r_jvel = JOINT_VEL_W * torch.sum(qvel * qvel, dim=-1)

    #     # Action rate penalty (encourages smooth actions)
    #     r_arate = ACT_RATE_W * torch.sum((action - last_action) ** 2, dim=-1)

    #     # Success bonus (one-time reward for reaching target)
    #     success = distance < SUCCESS_THRESHOLD
    #     r_success = success.to(distance.dtype) * SUCCESS_BONUS

    #     # Total reward
    #     reward = r_pos + r_fine + r_jvel + r_success + r_arate

    #     return reward

    def _is_success(self, obs: dict[str, Any]) -> Tensor:
        """Checks if the task is successful.

        Args:
            obs: The current observation dictionary.

        Returns:
            A boolean tensor of shape (B,) indicating success for each env.
        """
        tcp_pos_left = obs["state"]["left"]["gripper"]["tcp_pos"]
        tcp_pos_right = obs["state"]["right"]["gripper"]["tcp_pos"]
        target_pos_left = obs["state"]["other"]["target_left_pos"]
        target_pos_right = obs["state"]["other"]["target_right_pos"]

        distance_left = torch.norm(tcp_pos_left - target_pos_left, dim=-1)  # (B,)
        distance_right = torch.norm(tcp_pos_right - target_pos_right, dim=-1)  # (B,)
        is_left_within_success_threshold = distance_left < SUCCESS_THRESHOLD
        is_right_within_success_threshold = distance_right < SUCCESS_THRESHOLD
        return torch.bitwise_and(
            is_left_within_success_threshold,
            is_right_within_success_threshold,
        )

    def _is_failure(self, obs) -> torch.Tensor:
        return self._is_tcp_outside_workspace(obs)
