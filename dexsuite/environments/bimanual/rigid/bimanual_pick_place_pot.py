"""Bimanual pot pick-and-place rigid environment.

The robot must lift a pot from a burner on the hot plate and place it onto a
kitchen mat using both arms.
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
SUBSTEPS = 2
HORIZON = 7000

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

# Bimanual lift detection
GRAB_DIST = 0.10  # each TCP must be within this distance from pot AABB
LIFT_Z_ABOVE = 0.020  # pot bottom must rise this much above its spawn bottom

# Scene
HOT_PLATE_SCALE = 2
KITCHEN_MAT_SCALE = 0.8
POT_SCALE = 0.1

# Base poses
HOT_PLATE_Z = -0.01
MAT_Z = -0.005
POT_Z_OFFSET = 0.06  # Z offset for pot
HOT_PLATE_BASE = (0.5, -0.20, -0.01)
MAT_BASE = (HOT_PLATE_BASE[0], HOT_PLATE_BASE[1] + 0.55, -0.005)

# Sampling
BURNERS_POSITIONS = [(-0.07, 0.14), (-0.10, -0.17), (0.14, 0.17), (0.12, -0.15)]
BURNER_INDICES = range(len(BURNERS_POSITIONS))
BURNERS_TENSOR = torch.tensor(BURNERS_POSITIONS, dtype=torch.float32)

POT_HOTPLATE_SAMPLING = (0.01, 0.01)  # Pot is precise, small random band
YAW_DEG = 20.0

# Mat collision box dimensions
MAT_W, MAT_L, THICK = 0.44, 0.30, 0.010

# Target points for pot
POT_TARGET_LEFT_LOCAL = (0, 0.076, 0.028)
POT_TARGET_RIGHT_LOCAL = (0, -0.076, 0.028)


@register_env("bimanual_pick_place_pot")
class BimanualPickPlacePotEnv(BaseEnv):
    """Pick a pot and place it on the mat using both arms."""

    allowed_modes = ("bimanual",)

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
        # runtime counters / cached bounds
        self._resting = 0
        self._drops = 0
        self._mat_lo = self._mat_hi = self._mat_top_z = None
        self._inner_lo = self._inner_hi = None
        # "carried with two arms" detector
        self._spawn_bottom_z = None
        self._lifted_by_two = False

    # -------------------------- scene setup --------------------------
    def _setup_scene(self) -> None:
        hp_path = get_object_path("kitchen/hot_plate")
        mat_path = get_object_path("kitchen/kitchen_mat")
        pot_path = get_object_path("partnet/pot/mobility.urdf")  # PartNet pot

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

        # The pot itself (URDF, rigid)
        self.pot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(pot_path),
                collision=True,
                scale=POT_SCALE,
                euler=(0.0, 0.0, 0.0),
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Aluminium(),
        )

        self.pot_target_left = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.0075,
                collision=False,
                quat=(0.5, -0.5, 0.5, 0.5),
                fixed=True,
                pos=POT_TARGET_LEFT_LOCAL,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(255, 0.0, 0.0)),
        )
        link_entities(
            parent_entity=self.pot,
            child_entity=self.pot_target_left,
            parent_link_name=self.pot.links[0].name,
            child_link_name=self.pot_target_left.links[0].name,
        )

        self.pot_target_right = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.0075,
                collision=False,
                quat=(0.5, -0.5, 0.5, 0.5),
                fixed=True,
                pos=POT_TARGET_RIGHT_LOCAL,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(255, 0.0, 0.0)),
        )
        link_entities(
            parent_entity=self.pot,
            child_entity=self.pot_target_right,
            parent_link_name=self.pot.links[0].name,
            child_link_name=self.pot_target_right.links[0].name,
        )

    # ----------------------- episode lifecycle -----------------------
    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        self._resting = 0
        self._drops = 0
        self._lifted_by_two = False
        self._spawn_bottom_z = None

        # Pot randomization
        burner_index = sample_index(BURNER_INDICES, envs_idx)
        burner_offset_xy = BURNERS_TENSOR[burner_index.squeeze(-1)].to(self.device)
        hot_plate_origin_xy = torch.tensor(
            HOT_PLATE_BASE[:2],
            dtype=torch.float32,
            device=self.device,
        )
        burner_world_xy = hot_plate_origin_xy + burner_offset_xy

        pot_pos = sample_around_point_xy(
            anchor_xy=burner_world_xy,
            band_xy=POT_HOTPLATE_SAMPLING,  # Small band for pot
            z=float(HOT_PLATE_Z + POT_Z_OFFSET),
            mode="circle",
            env_idx=envs_idx,
        )
        self.pot.set_pos(pot_pos, envs_idx=envs_idx)

        # Yaw randomization
        yaw_half_range_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(
            yaw_range=(-yaw_half_range_rad, yaw_half_range_rad),
        )
        # Pot URDF is oriented correctly
        pot_world_quat = yaw_randomizer.quat(
            base_quat=None,
            frame="local",
            env_idx=envs_idx,
        )
        self.pot.set_quat(pot_world_quat, envs_idx=envs_idx)

        # Cache mat AABB (lo/hi on XY and top-Z)
        mat_lo, mat_hi = self.mat_collision.get_AABB()
        self.mat_lo, self.mat_hi = (
            mat_lo.to(self.device),
            mat_hi.to(
                self.device,
            ),
        )
        self._mat_top_z = float(self.mat_hi[2])
        self._inner_lo = self.mat_lo[:2] + XY_MARGIN
        self._inner_hi = self.mat_hi[:2] - XY_MARGIN

        # Remember spawn bottom-Z for "lifted" check
        pot_aabb = self.pot.get_AABB()
        self._spawn_bottom_z = float(pot_aabb[0, 2].item())

    # --------------------------- helpers -----------------------------
    @staticmethod
    def _tcp(obs, side: str) -> torch.Tensor:
        return obs["state"][side]["gripper"]["tcp_pos"]

    @staticmethod
    def _overlap_point_aabb(
        p: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> bool:
        return bool((p >= lo).all() and (p <= hi).all())

    @staticmethod
    def _dist_point_aabb(
        p: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> torch.Tensor:
        clamped = torch.minimum(torch.maximum(p, lo), hi)
        return torch.linalg.vector_norm(p - clamped)
