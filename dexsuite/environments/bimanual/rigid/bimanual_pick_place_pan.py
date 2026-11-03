"""Bimanual pan pick-and-place rigid environment.

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
    sample_around_point_xy,
    sample_index,
)
from dexsuite.utils.robot_utils import link_entities

# Simulation
SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 700

# Success / failure thresholds
XY_MARGIN = 0.04
Z_TOL = 0.02
TCP_CLEARANCE = 0.08
REST_FRAMES = 5
REST_LIN_V_MAX = 0.05
REST_ANG_V_MAX = 0.50
DROP_VZ_THR = -0.60
DROP_TCP_DIST = 0.20
DROP_GRACE = 3

# Scene
HOT_PLATE_SCALE = 2
KITCHEN_MAT_SCALE = 0.8

# Base poses
HOT_PLATE_Z = -0.01
MAT_Z = -0.005
PAN_Z_OFFSET = 0.01  # Z offset for pan relative to hot plate

HOT_PLATE_BASE = (0.65, -0.10, -0.01)
MAT_BASE = (HOT_PLATE_BASE[0], HOT_PLATE_BASE[1] + 0.55, -0.005)

# Sampling
BURNERS_POSITIONS = [(-0.07, 0.14), (-0.10, -0.17), (0.14, 0.17), (0.12, -0.15)]
BURNER_INDICES = range(len(BURNERS_POSITIONS))
BURNERS_TENSOR = torch.tensor(BURNERS_POSITIONS, dtype=torch.float32)

PAN_HOTPLATE_SAMPLING = (0.03, 0.03)
YAW_DEG = 20.0

GAP_MARGIN = 0.05

ZERO_POSITION = (0, 0, 0)

# Mat collision box dimensions
MAT_W, MAT_L, THICK = 0.44, 0.30, 0.010

# Target point for pan
PAN_TARGET_LOCAL = (0.0, 0.0535, 0.18)

MAT_OFFSET = 1


@register_env("bimanual_pick_place_pan")
class Bimanual_PickPlacePanEnv(BaseEnv):
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
        self._resting = 0
        self._drops = 0
        self._mat_lo = self._mat_hi = self._mat_top_z = None
        self._inner_lo = self._inner_hi = None

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



    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        self._resting = 0
        self._drops = 0

        # SET HOT PLATES AND MAT CLOSE TO CENTER OF AABB
        hot_plate_base = self.world_aabb.half("y", "low").center()
        hot_plate_base[0] *= 1.2
        hot_plate_base[2] = -0.01
        self.hot_plate.set_pos(hot_plate_base, envs_idx=envs_idx)
        mat_base = self.world_aabb.half("y", "high").center()
        mat_base[2] = -0.005
        self.kitchen_mat.set_pos(mat_base, envs_idx=envs_idx)
        mat_collision_base = (mat_base[0], mat_base[1], mat_base[2] + THICK * 0.5)
        self.mat_collision.set_pos(mat_collision_base, envs_idx=envs_idx)



        burner_index = sample_index(
            BURNER_INDICES,
            envs_idx,
        )  # get a random burner index i.e. choose a random burner to place the pan on
        burner_offset_xy = BURNERS_TENSOR[burner_index.squeeze(-1)].to(
            self.device,
        )  # make it shape (N,) instead of (N,1)
        hot_plate_center_xy = torch.tensor(
            hot_plate_base[:2],
            dtype=torch.float32,
            device=self.device,
        )  # get the hotplate's base XY as a tensor
        burner_world_xy = (
            hot_plate_center_xy + burner_offset_xy
        )  # position of the burner with the offset of the burners relative to the hotplate

        pan_pos = sample_around_point_xy(
            anchor_xy=burner_world_xy,  # Anchor to one of the hotplate's burners
            band_xy=PAN_HOTPLATE_SAMPLING,
            z=float(HOT_PLATE_Z + PAN_Z_OFFSET),
            mode="circle",
            env_idx=envs_idx,
        )


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
        self.pan.set_quat(pan_world_quat, envs_idx=envs_idx)  # flat + rnadom yaw

        # Get AABB from the collision box, not the visual mat
        mat_lo, mat_hi = self.mat_collision.get_AABB()
        self.mat_lo, self.mat_hi = mat_lo.to(self.device), mat_hi.to(self.device)

        # mat_top_z is the top surface of the collision box
        self._mat_top_z = float(self.mat_hi[2])

        # Get inner bounds for success
        self._inner_lo = self.mat_lo[:2] + XY_MARGIN
        self._inner_hi = self.mat_hi[:2] - XY_MARGIN

