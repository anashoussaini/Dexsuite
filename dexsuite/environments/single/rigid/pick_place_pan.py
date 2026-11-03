"""Single-arm pan pick-and-place rigid environment.

The robot must move a frying pan from a burner on the hot plate onto a kitchen
mat.
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
    sample_in_aabb_center,
    sample_index,
)
from dexsuite.utils.robot_utils import link_entities

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 500  # At 20hz this is 25 seconds

# Success thresholds
XY_MARGIN = 0.04
Z_TOL = 0.02
TCP_CLEARANCE = 0.08
REST_FRAMES = 5
REST_LIN_V_MAX = 0.05
REST_ANG_V_MAX = 0.50

# Scene
HOT_PLATE_SCALE = 2
KITCHEN_MAT_SCALE = 0.8

# Base poses
HOT_PLATE_Z = -0.01
PAN_Z_OFFSET = 0.01  # Z offset for pan relative to hot plate

HOT_PLATE_BASE = (0.65, -0.10, -0.01)
MAT_BASE = (HOT_PLATE_BASE[0], HOT_PLATE_BASE[1] + 0.55, -0.005)

# Sampling
BURNERS_POSITIONS = [(-0.07, 0.14), (-0.10, -0.17), (0.14, 0.17), (0.12, -0.15)]
BURNER_INDICES = range(len(BURNERS_POSITIONS))
BURNERS_TENSOR = torch.tensor(BURNERS_POSITIONS, dtype=torch.float32)

PAN_HOTPLATE_SAMPLING = (0.03, 0.03)
YAW_DEG = 20.0


# Mat collision box dimensions
MAT_W, MAT_L, THICK = 0.44, 0.30, 0.010

# Target point for pan
PAN_TARGET_LOCAL = (0.0, 0.0535, 0.18)


@register_env("pick_place_pan")
class PickPlacePanEnv(BaseEnv):
    """Pick and place a pan onto a kitchen mat."""

    def __init__(
        self,
        *,
        robot,
        cameras,
        sim,
        render_mode: str | None,
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
        hp_path = get_object_path("kitchen/hot_plate")
        mat_path = get_object_path("kitchen/kitchen_mat")
        pan_path = get_object_path("kitchen/pan")

        self.hot_plate = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(hp_path),
                scale=HOT_PLATE_SCALE,
                pos=HOT_PLATE_BASE,
                euler=(0.0, 0.0, -90.0),
                collision=False,
                visualization=True,
            ),
            material=gs.materials.Rigid(),
        )

        self.kitchen_mat = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(mat_path),
                scale=KITCHEN_MAT_SCALE,
                pos=MAT_BASE,
                euler=(90.0, 0.0, 0.0),
                collision=False,
                visualization=True,
            ),
        )

        self.mat_collision = self.scene.add_entity(
            gs.morphs.Box(
                size=(MAT_W, MAT_L, THICK),
                pos=(MAT_BASE[0], MAT_BASE[1], MAT_BASE[2] + THICK * 0.5),
                euler=(0.0, 0.0, 0.0),
                collision=True,
                visualization=False,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
        )

        self.pan = self.scene.add_entity(
            gs.morphs.MJCF(file=str(pan_path), collision=True, pos=HOT_PLATE_BASE),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Aluminium(),
        )

        self.pan_target = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.01,
                collision=False,
                quat=(0.5, -0.5, 0.5, 0.5),
                fixed=True,
                pos=PAN_TARGET_LOCAL,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(255, 0.0, 0.0)),
        )

        link_entities(
            parent_entity=self.pan,
            child_entity=self.pan_target,
            parent_link_name=self.pan.links[0].name,
            child_link_name=self.pan_target.links[0].name,
        )

    def _on_episode_start(self, envs_idx: None) -> None:
        hot_plate_base = sample_in_aabb_center(
            self.world_aabb,
            env_idx=envs_idx,
        ).clone()
        hot_plate_base[...,0] *=1.2
        hot_plate_base[...,1] -= 0.2
        hot_plate_base[...,2] = -0.01
        self.hot_plate.set_pos(hot_plate_base,envs_idx = envs_idx)
        mat_base = hot_plate_base.clone()
        mat_base[...,1] += 0.55
        mat_base[...,2] = -0.005
        self.kitchen_mat.set_pos(mat_base, envs_idx = envs_idx)
        mat_collision_base = mat_base.clone()
        mat_collision_base[...,2] += THICK * 0.5
        self.mat_collision.set_pos(mat_collision_base,envs_idx = envs_idx)

        rand_burner_index = sample_index(BURNER_INDICES, envs_idx).to("cpu") #get a random burner index i.e. choose a random burner to place the pan on
        burner_offset_xy = BURNERS_TENSOR[rand_burner_index.squeeze(-1)].to(self.device) # make it shape (N,) instead of (N,1)
        hot_plate_base_xy = torch.tensor(hot_plate_base[...,:2], dtype=torch.float32, device=self.device) # get the hotplate's base XY as a tensor
        burner_world_xy = (hot_plate_base_xy + burner_offset_xy).reshape(-1, 2) #position of the burner with the offset of the burners relative to the hotplate
        pan_pos =torch.cat([burner_world_xy, torch.full((burner_world_xy.shape[0], 1), float(HOT_PLATE_Z + PAN_Z_OFFSET), device=burner_world_xy.device, dtype=burner_world_xy.dtype)], dim=-1)  # pan position is burner position with Z offset
        if envs_idx is None and self.n_envs == 1:
            pan_pos = pan_pos.squeeze(0)


        yaw_half_range_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(
            yaw_range=(-yaw_half_range_rad, yaw_half_range_rad),
        )

        self.pan.set_pos(pan_pos, envs_idx=envs_idx)

        pan_flat_base_quat = (
            0.5,
            0.5,
            -0.5,
            -0.5,
        )  # Make the pan lay flat (default is vertical)

        pan_world_quat = yaw_randomizer.quat(
            base_quat=pan_flat_base_quat,
            frame="local",
            env_idx=envs_idx,
        )  # random yaw

        self.pan.set_quat(pan_world_quat, envs_idx=envs_idx)  # flat + random yaw

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        pan_aabb = self.pan.get_AABB()
        mat_aabb = self.mat_collision.get_AABB()
        if mat_aabb.ndim == 2:
            mat_top_z = mat_aabb[1, 2:3]  # (1,) to avoid scalar () tensors
        else:
            mat_top_z = mat_aabb[..., 1, 2]
        return {
            "pan_pos": self.pan.get_pos(),
            "pan_quat": self.pan.get_quat(),
            "pan_vel": self.pan.get_vel(),
            "pan_ang": self.pan.get_ang(),
            "pan_aabb_lo": pan_aabb[..., 0, :],
            "pan_aabb_hi": pan_aabb[..., 1, :],
            "mat_inner_lo": mat_aabb[..., 0, :2] + XY_MARGIN,
            "mat_inner_hi": mat_aabb[..., 1, :2] - XY_MARGIN,
            "mat_top_z": mat_top_z,
        }

    def _is_success(self, obs) -> torch.Tensor:
        """Return True when the pan has rested on the mat for REST_FRAMES steps."""
        other = obs["state"]["other"]
        pan_pos = other["pan_pos"]
        pan_lo = other["pan_aabb_lo"]

        inner_lo = other["mat_inner_lo"]
        inner_hi = other["mat_inner_hi"]
        mat_top_z = other["mat_top_z"]

        xy_ok = (
            (pan_pos[..., 0] >= inner_lo[..., 0])
            & (pan_pos[..., 0] <= inner_hi[..., 0])
            & (pan_pos[..., 1] >= inner_lo[..., 1])
            & (pan_pos[..., 1] <= inner_hi[..., 1])
        )
        z_ok = torch.abs(pan_lo[..., 2] - mat_top_z) <= Z_TOL

        return z_ok & xy_ok
