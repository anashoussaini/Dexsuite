"""Single-arm fruit pick-and-place rigid environment.

The robot must place an apple into a bowl and leave it resting inside the bowl
region.
"""

from __future__ import annotations

import math
from typing import Any

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.randomizers import (
    YawRandomizer,
    sample_in_aabb_center_xyz_band,
    sample_xy_k_noncolliding,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 400  # At 20hz this is 20 seconds

# Success thresholds
XY_MARGIN = 0.04
Z_TOL = 0.01


# Assets
bowl_PATH = "kitchen/bowl"
apple_PATH = "food/apple"
lemon_PATH = "food/lemon"
# Resting checks
REST_FRAMES = 5
REST_VEL_THR = 0.05
REST_ANG_THR = 0.50

# Sampling
bowl_Z = -0.025
fruit_Z = 0.03
bowl_XY_BAND = 0.12
fruit_XY_BAND = 0.12
YAW_DEG = 10.0
MIN_SEP = 0.15
MAX_SEP = 0.45

@register_env("pick_place_fruit")
class PickPlaceappleEnv(BaseEnv):
    """Pick a apple and place it into a dish bowl."""

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
        """Initialize the PickPlaceappleEnv environment."""
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
        bowl_path = get_object_path(bowl_PATH)
        apple_path = get_object_path(apple_PATH)
        lemon_path = get_object_path(lemon_PATH)

        self.bowl = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(bowl_path),
                scale=1.7,
                decimate = False,
                euler=(0.0, 0.0, 0.0),
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Plastic(
                color=(0.95, 0.95, 0.95),
                roughness=0.35,
                metallic=0.0,
                ior=1.47,
            ),
        )

        self.apple = self.scene.add_entity(
            morph=gs.morphs.MJCF(file=str(apple_path)),
            material=gs.materials.Rigid(),
        )
        self.lemon = self.scene.add_entity(
            morph=gs.morphs.MJCF(file=str(lemon_path),
                                 scale=1.3,
                                 ),
            material=gs.materials.Rigid(),
            )
    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:


        bowl_box = self.world_aabb.half("y", "high")
        fruit_box = self.world_aabb.half("y", "low")
        bowl_pos = sample_in_aabb_center_xyz_band(
            bowl_box,
            band_xyz=(bowl_XY_BAND, bowl_XY_BAND, 0.0),
            range_z=(bowl_Z, bowl_Z),
            env_idx=envs_idx,
        )



        apple_pos, lemon_pose = sample_xy_k_noncolliding(
            fruit_box,
            k=2,
            band_xy=(fruit_XY_BAND, fruit_XY_BAND, 0.0),
            z=fruit_Z,
            min_sep=MIN_SEP,
            max_sep=MAX_SEP,
            env_idx=envs_idx,
            mode="square",
            max_tries=32,
        )

        self.bowl.set_pos(bowl_pos, envs_idx=envs_idx)
        self.apple.set_pos(apple_pos, envs_idx=envs_idx)
        self.lemon.set_pos(lemon_pose, envs_idx=envs_idx)

        yaw_half_range_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(
            yaw_range=(-yaw_half_range_rad, yaw_half_range_rad),
        )
        bowl_quat = yaw_randomizer.quat(env_idx=envs_idx)
        apple_quat = yaw_randomizer.quat(env_idx=envs_idx)
        lemon_quat = yaw_randomizer.quat(env_idx=envs_idx)
        self.bowl.set_quat(bowl_quat, envs_idx=envs_idx, relative=True)
        self.apple.set_quat(apple_quat, envs_idx=envs_idx, relative=True)
        self.lemon.set_quat(lemon_quat, envs_idx=envs_idx, relative=True)

    def _get_extra_obs(self) -> dict[str, Any]:
        bowl_aabb = self.bowl.get_AABB()
        if bowl_aabb.ndim == 2:
            bowl_aabb_min, bowl_aabb_max = bowl_aabb[0], bowl_aabb[1]
            bowl_top_z = bowl_aabb_max[2:3]  # (1,) to avoid scalar () tensors
        else:
            bowl_aabb_min, bowl_aabb_max = bowl_aabb[:, 0, :], bowl_aabb[:, 1, :]
            bowl_top_z = bowl_aabb_max[..., 2]

        return {
            "bowl_pos": self.bowl.get_pos(),
            "bowl_quat": self.bowl.get_quat(),
            "bowl_aabb_lo": bowl_aabb_min,
            "bowl_aabb_hi": bowl_aabb_max,
            "bowl_inner_lo": bowl_aabb_min[..., :2] + XY_MARGIN,
            "bowl_inner_hi": bowl_aabb_max[..., :2] - XY_MARGIN,
            "bowl_top_z": bowl_top_z,
            "apple_pos": self.apple.get_pos(),
            "apple_quat": self.apple.get_quat(),
            "apple_vel": self.apple.get_vel(),
            "apple_ang": self.apple.get_ang(),
            "lemon_pos": self.lemon.get_pos(),
            "lemon_quat": self.lemon.get_quat(),
            "lemon_vel": self.lemon.get_vel(),
            "lemon_ang": self.lemon.get_ang(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Return True when the apple has rested in the bowl for REST_FRAMES steps."""
        other = obs["state"]["other"]
        apple_pos = other["apple_pos"].reshape(-1, 3)
        lemon_pos = other["lemon_pos"].reshape(-1, 3)
        tcp = obs["state"]["gripper"]["tcp_pos"].reshape(-1, 3)

        inner_lo = other["bowl_inner_lo"].reshape(-1, 2)
        inner_hi = other["bowl_inner_hi"].reshape(-1, 2)
        bowl_lo = other["bowl_aabb_lo"].reshape(-1, 3)
        bowl_hi = other["bowl_aabb_hi"].reshape(-1, 3)
        bowl_top_z = other["bowl_top_z"].reshape(-1)

        apple_inside_xy = (
            (apple_pos[:, 0] >= inner_lo[:, 0])
            & (apple_pos[:, 0] <= inner_hi[:, 0])
            & (apple_pos[:, 1] >= inner_lo[:, 1])
            & (apple_pos[:, 1] <= inner_hi[:, 1])
        )
        lemon_inside_xy = (
            (lemon_pos[:, 0] >= inner_lo[:, 0])
            & (lemon_pos[:, 0] <= inner_hi[:, 0])
            & (lemon_pos[:, 1] >= inner_lo[:, 1])
            & (lemon_pos[:, 1] <= inner_hi[:, 1])
        )
        z_ok_apple = (bowl_top_z  - apple_pos[:, 2]  >= Z_TOL) & (apple_pos[:,2] > 0)
        z_ok_lemon = (bowl_top_z  - lemon_pos[:, 2]  >= Z_TOL) & (lemon_pos[:,2] > 0)
        tcp_inside_bowl = (
            (tcp[:, 0] >= bowl_lo[:, 0])
            & (tcp[:, 0] <= bowl_hi[:, 0])
            & (tcp[:, 1] >= bowl_lo[:, 1])
            & (tcp[:, 1] <= bowl_hi[:, 1])
            & (tcp[:, 2] >= bowl_lo[:, 2])
            & (tcp[:, 2] <= bowl_hi[:, 2])
        )
        tcp_clear = ~tcp_inside_bowl


        return tcp_clear & apple_inside_xy & lemon_inside_xy & z_ok_apple & z_ok_lemon
