"""Single-arm mug pick-and-place rigid environment.

The robot must place a mug into a dish rack and leave it resting inside the
rack region.
"""

from __future__ import annotations

import math
from typing import Any

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.randomizers import YawRandomizer, sample_in_aabb_center_xyz_band

# Simulation
SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 400  # At 20hz this is 20 seconds

# Success thresholds
XY_MARGIN = 0.04
Z_TOL = 0.05

# Assets
RACK_PATH = "kitchen/dish_rack"
MUG_PATH = "kitchen/mug"

# Resting checks
REST_FRAMES = 5
REST_VEL_THR = 0.05
REST_ANG_THR = 0.50

# Sampling
RACK_Z = 0.085
MUG_Z = 0.05
RACK_XY_BAND = 0.12
MUG_XY_BAND = 0.12
YAW_DEG = 10.0


@register_env("pick_place_mug")
class PickPlaceMugEnv(BaseEnv):
    """Pick a mug and place it into a dish rack."""

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
        """Initialize the PickPlaceMugEnv environment."""
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
        rack_path = get_object_path(RACK_PATH)
        mug_path = get_object_path(MUG_PATH)

        self.rack = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(rack_path),
                batch_fixed_verts=True,  # per-env poses of this fixed model in batched resets
                scale=0.8,
                euler=(90.0, 0.0, 0.0),
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Plastic(
                color=(0.95, 0.95, 0.95),
                roughness=0.35,
                metallic=0.0,
                ior=1.47,
            ),
        )

        self.mug = self.scene.add_entity(
            morph=gs.morphs.MJCF(file=str(mug_path)),
            material=gs.materials.Rigid(),
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:


        rack_box = self.world_aabb.half("y", "high")
        mug_box = self.world_aabb.half("y", "low")
        rack_pos = sample_in_aabb_center_xyz_band(
            rack_box,
            band_xyz=(RACK_XY_BAND, RACK_XY_BAND, 0.0),
            range_z=(RACK_Z, RACK_Z),
            env_idx=envs_idx,
        )

        mug_pos = sample_in_aabb_center_xyz_band(
            mug_box,
            band_xyz=(MUG_XY_BAND, MUG_XY_BAND, 0.0),
            range_z=(MUG_Z, MUG_Z),
            env_idx=envs_idx,
        )

        self.rack.set_pos(rack_pos, envs_idx=envs_idx)
        self.mug.set_pos(mug_pos, envs_idx=envs_idx)

        yaw_half_range_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(
            yaw_range=(-yaw_half_range_rad, yaw_half_range_rad),
        )
        rack_quat = yaw_randomizer.quat(env_idx=envs_idx)
        mug_quat = yaw_randomizer.quat(env_idx=envs_idx)
        self.rack.set_quat(rack_quat, envs_idx=envs_idx, relative=True)
        self.mug.set_quat(mug_quat, envs_idx=envs_idx, relative=True)

    def _get_extra_obs(self) -> dict[str, Any]:
        rack_aabb = self.rack.get_AABB()
        if rack_aabb.ndim == 2:
            rack_aabb_min, rack_aabb_max = rack_aabb[0], rack_aabb[1]
            rack_top_z = rack_aabb_max[2:3]  # (1,) to avoid scalar () tensors
        else:
            rack_aabb_min, rack_aabb_max = rack_aabb[:, 0, :], rack_aabb[:, 1, :]
            rack_top_z = rack_aabb_max[..., 2]

        return {
            "rack_pos": self.rack.get_pos(),
            "rack_quat": self.rack.get_quat(),
            "rack_aabb_lo": rack_aabb_min,
            "rack_aabb_hi": rack_aabb_max,
            "rack_inner_lo": rack_aabb_min[..., :2] + XY_MARGIN,
            "rack_inner_hi": rack_aabb_max[..., :2] - XY_MARGIN,
            "rack_top_z": rack_top_z,
            "mug_pos": self.mug.get_pos(),
            "mug_quat": self.mug.get_quat(),
            "mug_vel": self.mug.get_vel(),
            "mug_ang": self.mug.get_ang(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Return True when the mug has rested in the rack for REST_FRAMES steps."""
        other = obs["state"]["other"]
        mug_pos = other["mug_pos"].reshape(-1, 3)
        tcp = obs["state"]["gripper"]["tcp_pos"].reshape(-1, 3)

        inner_lo = other["rack_inner_lo"].reshape(-1, 2)
        inner_hi = other["rack_inner_hi"].reshape(-1, 2)
        rack_lo = other["rack_aabb_lo"].reshape(-1, 3)
        rack_hi = other["rack_aabb_hi"].reshape(-1, 3)
        rack_top_z = other["rack_top_z"].reshape(-1)

        inside_xy = (
            (mug_pos[:, 0] >= inner_lo[:, 0])
            & (mug_pos[:, 0] <= inner_hi[:, 0])
            & (mug_pos[:, 1] >= inner_lo[:, 1])
            & (mug_pos[:, 1] <= inner_hi[:, 1])
        )
        z_ok = (rack_top_z  - mug_pos[:, 2]  >= Z_TOL) & (mug_pos[:,2] > 0)
        tcp_inside_rack = (
            (tcp[:, 0] >= rack_lo[:, 0])
            & (tcp[:, 0] <= rack_hi[:, 0])
            & (tcp[:, 1] >= rack_lo[:, 1])
            & (tcp[:, 1] <= rack_hi[:, 1])
            & (tcp[:, 2] >= rack_lo[:, 2])
            & (tcp[:, 2] <= rack_hi[:, 2])
        )
        tcp_clear = ~tcp_inside_rack


        return tcp_clear & inside_xy & z_ok
