"""Single-arm lift baseline environment.

The robot must lift a rigid cube from the table to a fixed goal marker above
its spawn position.
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
HORIZON = 600

# Task
CUBE_SIZE = 0.05
GOAL_R = 0.02
GOAL_UP = 0.20 + CUBE_SIZE / 2.0
LIFT_EPS = 0.01

# Reward shaping
CENTER_BAND = 0.25
TCP_THR = 0.05
STD = 0.3

REACH_STD = 0.30
REACH_FINE_STD = 0.08
GOAL_STD = 0.30
GOAL_FINE_STD = 0.05
GRASP_HEIGHT_OFFSET = 0.018
MIN_LIFT_HEIGHT = 0.04

REACH_W = 0.25
REACH_FINE_W = 2.0
LIFTED_W = 15.0
GOAL_TRACKING_W = 16.0
GOAL_TRACKING_FINE_W = 5.0
NEAR_CLOSE_W = 2.0

ACT_RATE_W = -1.0e-4  # Isaac Lab-style action_rate penalty on ||a - a_prev||^2
JOINT_VEL_W = -1.0e-4  # joint_vel penalty on ||qdot||^2
SUCCESS_THRESHOLD = 0.03  # 3 cm success, like your docstring
TIME_PENALTY = 2.0e-3


def _as_feature_batch(value: torch.Tensor) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor.reshape(-1, tensor.shape[-1])


@register_env("lift")
class LiftEnv(BaseEnv):
    """Lift a cube to a target height above its spawn."""

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
            surface=gs.surfaces.Default(color=(0.90, 0.76, 0.16)),
        )
        self.goal = self.scene.add_entity(
            gs.morphs.Sphere(radius=GOAL_R, fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        """Reset cube pose and place the goal above the cube.

        Args:
            envs_idx: Optional subset of environments to reset (batched sim).
        """
        # cube at table level
        cube_pos = sample_in_aabb_center_xy_band(
            self.world_aabb,
            band_xy=(0.1, 0.25),
            z=CUBE_SIZE / 2.0,
            env_idx=envs_idx,
            mode="square",
        )
        self.cube.set_pos(cube_pos, envs_idx=envs_idx)
        goal_pos = cube_pos + torch.as_tensor(
            [0.0, 0.0, GOAL_UP],
            device=cube_pos.device,
            dtype=cube_pos.dtype,
        )
        self.goal.set_pos(goal_pos, envs_idx=envs_idx)

    def _compute_reward(self, obs: dict) -> torch.Tensor:
        """Compute an Isaac Lab-style dense lift reward.

        Args:
            obs: Observation dict produced by BaseEnv.step.

        Returns:
            Reward tensor of shape (n_envs,).
        """
        terms = self._lift_reward_terms(obs)

        action = _as_feature_batch(obs["state"]["other"]["action"])
        last_action = _as_feature_batch(obs["state"]["other"]["last_action"])
        action_rate = torch.sum((action - last_action) ** 2, dim=-1)
        joint_velocity = _as_feature_batch(obs["state"]["manipulator"]["qvel"])

        reward = REACH_W * terms["reach_score"]
        reward += REACH_FINE_W * terms["reach_fine_score"]
        reward += LIFTED_W * terms["lifted_score"]
        reward += GOAL_TRACKING_W * terms["goal_score"]
        reward += GOAL_TRACKING_FINE_W * terms["goal_fine_score"]
        reward += NEAR_CLOSE_W * terms["near_close_score"]
        reward += JOINT_VEL_W * torch.sum(joint_velocity**2, dim=-1)
        reward += ACT_RATE_W * action_rate
        reward -= TIME_PENALTY
        return reward

    def _lift_reward_terms(self, obs: dict) -> dict[str, torch.Tensor]:
        """Return shaped reward terms for diagnostics and reward computation."""
        tcp_position = _as_feature_batch(obs["state"]["gripper"]["tcp_pos"])
        cube_position = _as_feature_batch(obs["state"]["other"]["cube_pos"])
        target_position = _as_feature_batch(obs["state"]["other"]["target_pos"])

        grasp_position = cube_position.clone()
        grasp_position[..., 2] += GRASP_HEIGHT_OFFSET

        tcp_to_grasp = torch.linalg.norm(tcp_position - grasp_position, dim=-1)
        reach_score = 1.0 - torch.tanh(tcp_to_grasp / REACH_STD)
        reach_fine_score = 1.0 - torch.tanh(tcp_to_grasp / REACH_FINE_STD)

        target_height = torch.clamp(
            target_position[..., 2] - CUBE_SIZE / 2.0,
            min=CUBE_SIZE / 2.0 + 1e-6,
        )
        height_score = torch.clamp(
            (cube_position[..., 2] - CUBE_SIZE / 2.0)
            / torch.clamp(target_height - CUBE_SIZE / 2.0, min=1e-6),
            min=0.0,
            max=1.0,
        )

        lifted_score = (cube_position[..., 2] > (CUBE_SIZE / 2.0 + MIN_LIFT_HEIGHT)).to(
            cube_position.dtype
        )
        cube_to_goal = torch.linalg.norm(cube_position - target_position, dim=-1)
        goal_score = lifted_score * (1.0 - torch.tanh(cube_to_goal / GOAL_STD))
        goal_fine_score = lifted_score * (
            1.0 - torch.tanh(cube_to_goal / GOAL_FINE_STD)
        )

        action = _as_feature_batch(obs["state"]["other"]["action"])
        close_score = torch.zeros_like(reach_score)
        if action.shape[-1] >= 9:
            close_score = torch.clamp(
                (action[..., 8] - action[..., 7]) * 0.5,
                min=0.0,
                max=1.0,
            )
        near_close_score = reach_fine_score * close_score
        return {
            "reach_score": reach_score,
            "reach_fine_score": reach_fine_score,
            "close_score": close_score,
            "near_close_score": near_close_score,
            "height_score": height_score,
            "lifted_score": lifted_score,
            "goal_score": goal_score,
            "goal_fine_score": goal_fine_score,
            "tcp_to_grasp": tcp_to_grasp,
            "cube_to_goal": cube_to_goal,
        }

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        """Expose cube and goal poses in obs['state']['other']."""
        return {
            "cube_pos": self.cube.get_pos(),
            "cube_quat": self.cube.get_quat(),
            "target_pos": self.goal.get_pos(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Check success based on cube proximity to the target pose."""
        cube_position = _as_feature_batch(obs["state"]["other"]["cube_pos"])
        target_position = _as_feature_batch(obs["state"]["other"]["target_pos"])
        # Success if cube height reaches the target height (with tolerance).
        target_z = GOAL_UP - LIFT_EPS
        is_height_reached = cube_position[..., 2] >= target_z
        xy_error = torch.linalg.norm(
            cube_position[..., :2] - target_position[..., :2],
            dim=-1,
        )

        # success if xy distance is below 5cm
        is_xy_aligned = xy_error <= 0.05
        return is_xy_aligned & is_height_reached

    def _is_failure(self, obs) -> torch.Tensor:
        """Fail if the cube falls below the table, matching Isaac Lab lift."""
        cube_position = _as_feature_batch(obs["state"]["other"]["cube_pos"])
        return cube_position[..., 2] < -0.05
