"""Single-arm drill-to-point rigid environment.

The robot must align a power-drill tip with a target marker on a wooden wall.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.orientation_utils import quat_to_rpy_wxyz_torch
from dexsuite.utils.randomizers import (
    YawRandomizer,
    sample_in_aabb_center_xyz_band,
)
from dexsuite.utils.robot_utils import link_entities

SIM_DT = 0.01
SUBSTEPS = 1
HORIZON = 300

DRILL_BASE_QUAT = (0.0, 0.0, 0.7071068, 0.7071068)

WALL_BASE_POS = (0.55, 0.55, 0.35)
WALL_EULER_DEG = (0.0, 0.0, 80.0)
WALL_SCALE = 0.5

WALL_BAND_XYZ = (0.05, 0.0, 0.05)
SLOPE_TILT = -35.0


DRILL_TIP_LOCAL = (-0.12, 0.055, 0.00)
TARGET_LOCAL_POS = (-0.02, 0.005, 0.00)


DRILL_YAW_DEG = 10.0


TIP_TARGET_DIST_THR = 0.03

PITCH_TARGET_RAD = 0.0
PITCH_TOL_RAD = math.radians(20.0)
YAW_TARGET_RAD = 1.39
YAW_TOL_RAD = math.radians(20.0)


@register_env("drill_to_point")
class DrillToPointEnv(BaseEnv):

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
        drill_path = get_object_path("tools/power_drill")
        wood_block = get_object_path("tools/wood_block/wood_block_fixed.xml")

        self.robot_offset_box = self.world_aabb.half("x", "low")

        self.wood_box = self.world_aabb.half("y", "high")
        self.wood_box.min[0] = self.robot_offset_box.max[0]

        self.drill_box = self.world_aabb.half("y", "low")
        self.drill_box.min[0] = self.robot_offset_box.max[0]
        drill_spawn_center_xyz = self.drill_box.center()
        drill_spawn_pos_list = drill_spawn_center_xyz.tolist()
        drill_spawn_pos_list[2] = 0.1
        self.drill_spawn_pos = tuple(drill_spawn_pos_list)
        self.drill_yaw_half_range_rad = math.radians(DRILL_YAW_DEG)

        self.drill = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(drill_path),
                quat=DRILL_BASE_QUAT,
                collision=True,
                pos=self.drill_spawn_pos,
            ),
            material=gs.materials.Rigid(),
        )

        self.drill_tip = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.01,
                collision=False,
                fixed=True,
                pos=DRILL_TIP_LOCAL,
                euler=(90,0,180),       # rotation to make axis coherent with world frame
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2)),
        )
        link_entities(
            parent_entity=self.drill,
            child_entity=self.drill_tip,
            parent_link_name=self.drill.links[0].name,
            child_link_name=self.drill_tip.links[0].name,
        )

        self.wooden_wall = self.scene.add_entity(
            gs.morphs.MJCF(
                batch_fixed_verts=True,  # per-env poses of this fixed model in batched resets
                file=str(wood_block),
                pos=WALL_BASE_POS,
                euler=WALL_EULER_DEG,
                scale=WALL_SCALE,
            ),
            material=gs.materials.Rigid(),
        )

        self.target = self.scene.add_entity(
            gs.morphs.Cylinder(
                height=0.01,
                radius=0.005,
                pos=TARGET_LOCAL_POS,
                # euler=TARGET_EULER_DEG,
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2)),
        )
        link_entities(
            parent_entity=self.wooden_wall,
            child_entity=self.target,
            parent_link_name=self.wooden_wall.links[0].name,
            child_link_name=self.target.links[0].name,
        )

    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        self._drops = 0
        # Randomize drill yaw around base orientation (absolute, world-frame)
        
        yaw_rand_quat = YawRandomizer( yaw_range=(-self.drill_yaw_half_range_rad, self.drill_yaw_half_range_rad)).quat(
            base_quat=DRILL_BASE_QUAT, frame="world",env_idx = envs_idx
        )
     
        self.drill.set_quat(yaw_rand_quat, envs_idx=envs_idx)

        slope_tilt_rad = math.radians(SLOPE_TILT)
        wall_slope = float(math.tan(slope_tilt_rad))
        wall_intercept = float(WALL_BASE_POS[1]) - wall_slope * float(WALL_BASE_POS[0])

        wall_position_sample = sample_in_aabb_center_xyz_band(
            self.wood_box,
            band_xyz=WALL_BAND_XYZ,
            range_z=(0.48, 0.58),
            env_idx=envs_idx,
        )
        is_single_env_sample = wall_position_sample.ndim == 1
        wall_positions = wall_position_sample.reshape(-1, 3).to(
            device=self.device,
            dtype=torch.float32,
        )

        wall_x = wall_positions[:, 0]
        wall_y_on_line = wall_slope * wall_x + float(wall_intercept)
        wall_y = wall_y_on_line.clamp(
            float(self.wood_box.min[1]),
            float(self.wood_box.max[1]),
        )
        wall_z = wall_positions[:, 2]

        clamped_wall_positions = torch.stack([wall_x, wall_y, wall_z], dim=1)
        clamped_wall_positions = self.wood_box.clamp(clamped_wall_positions)
        wall_positions_to_set = (
            clamped_wall_positions.squeeze(0)
            if is_single_env_sample
            else clamped_wall_positions
        )

        self.wooden_wall.set_pos(wall_positions_to_set, envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        """Expose task-relevant rigid-body poses under obs['state']['other'].

        Returns:
            A dictionary of tensors:
            - drill_pos, drill_quat
            - drill_tip_pos, drill_tip_quat
            - target_pos, target_quat
        """
        return {
            "drill_pos": self.drill.get_pos(),
            "drill_quat": self.drill.get_quat(),
            "drill_tip_pos": self.drill_tip.get_pos(),
            "drill_tip_quat": self.drill_tip.get_quat(),
            "target_pos": self.target.get_pos(),
            "target_quat": self.target.get_quat(),
        }


    def _is_success(self, obs) -> torch.Tensor:
        drill_tip_pos = obs["state"]["other"]["drill_tip_pos"].reshape(-1, 3)
        target_pos = obs["state"]["other"]["target_pos"].reshape(-1, 3)
        dist = torch.linalg.norm(drill_tip_pos - target_pos, dim=-1)

        dist_ok = dist <= TIP_TARGET_DIST_THR

        drill_quat = obs["state"]["other"]["drill_tip_quat"].reshape(-1, 4)
        rpy = quat_to_rpy_wxyz_torch(drill_quat)

        pitch = rpy[:, 1]
        yaw = rpy[:, 2]

        pitch_err = torch.abs(
            torch.atan2(
                torch.sin(pitch - PITCH_TARGET_RAD),
                torch.cos(pitch - PITCH_TARGET_RAD),
            ),
        )

        yaw_err = torch.abs(
            torch.atan2(
                torch.sin(yaw - YAW_TARGET_RAD),
                torch.cos(yaw - YAW_TARGET_RAD),
            ),
        )

        pitch_ok = pitch_err <= PITCH_TOL_RAD
        yaw_ok = yaw_err <= YAW_TOL_RAD

        return dist_ok & pitch_ok & yaw_ok
