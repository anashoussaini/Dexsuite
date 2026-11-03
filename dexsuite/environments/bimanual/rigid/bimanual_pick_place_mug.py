"""Bimanual mug pick-and-place rigid environment.

The robot must place a mug into a dish rack and leave it resting inside the
rack region.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.randomizers import (
    YawRandomizer,
    sample_in_aabb_center_xy_band,
)

# Simulation
SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 40_000

# Success thresholds
XY_MARGIN = 0.04
Z_TOL = 0.14

# Assets
RACK_PATH = "kitchen/dish_rack"
MUG_PATH = "kitchen/mug"

# Resting checks
REST_FRAMES = 5
REST_VEL_THR = 0.05
REST_ANG_THR = 0.50

# Failure (drop) checks
DROP_VZ_THR = -0.60
DROP_TCP_DIST_THR = 0.20
DROP_GRACE_FRAMES = 3

# Sampling
WS_MARGIN = (0.02, 0.02, 0.00)
RACK_Z = 0.085
MUG_Z = 0.05
RACK_XY_BAND = 0.12
MUG_XY_BAND = 0.12
YAW_DEG = 10.0


@register_env("bimanual_pick_place_mug")
class BimanualPickPlaceMugEnv(BaseEnv):
    """Pick a mug and place it into a dish rack."""

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
        rack_box = self._workspace_aabbs["left"].half("x", "high")
        mug_box = self._workspace_aabbs["right"].half("x", "high")

        rack_pos = sample_in_aabb_center_xy_band(
            rack_box,
            band_xy=(RACK_XY_BAND, RACK_XY_BAND, 0.0),
            z=RACK_Z,
            env_idx=envs_idx,
        )

        mug_pos = sample_in_aabb_center_xy_band(
            mug_box,
            band_xy=(MUG_XY_BAND, MUG_XY_BAND, 0.0),
            z=MUG_Z,
            env_idx=envs_idx,
        )

        self.rack.set_pos(pos=rack_pos, envs_idx=envs_idx)
        self.mug.set_pos(pos=mug_pos, envs_idx=envs_idx)

        yaw_half_range_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(
            yaw_range=(-yaw_half_range_rad, yaw_half_range_rad),
        )
        rack_quat = yaw_randomizer.quat(env_idx=envs_idx)
        mug_quat = yaw_randomizer.quat(env_idx=envs_idx)

        self.rack.set_quat(rack_quat, envs_idx=envs_idx, relative=True)
        self.mug.set_quat(mug_quat, envs_idx=envs_idx, relative=True)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        return {
            "rack_pos": self.rack.get_pos(),
            "rack_quat": self.rack.get_quat(),
            "mug_pos": self.mug.get_pos(),
            "mug_quat": self.mug.get_quat(),
            "mug_linear_vel": self.mug.get_vel(),
            "mug_angular_vel": self.mug.get_ang(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        mug_pos = obs["state"]["other"]["mug_pos"]
        rack_pos = obs["state"]["other"]["rack_pos"]

        mug_xy = mug_pos[..., :2]
        rack_xy = rack_pos[..., :2]

        xy_ok = torch.all(
            torch.abs(mug_xy - rack_xy) <= 0.1,
            dim=-1,
        )

        # 2 - Mug COM Z is within +/- 0.14 m of the rack top Z (no explicit contact check).
        mug_z = mug_pos[..., 2]
        rack_top_z = rack_pos[..., 2]
        z_ok = torch.abs(mug_z - rack_top_z) <= Z_TOL

        # 3 - Mug is "resting": low linear and angular speeds.
        mug_linear_vel = obs["state"]["other"]["mug_linear_vel"]
        mug_angular_vel = obs["state"]["other"]["mug_angular_vel"]
        vel_ok = torch.norm(mug_linear_vel, dim=-1) <= REST_VEL_THR
        ang_ok = torch.norm(mug_angular_vel, dim=-1) <= REST_ANG_THR

        return xy_ok & z_ok & vel_ok & ang_ok
