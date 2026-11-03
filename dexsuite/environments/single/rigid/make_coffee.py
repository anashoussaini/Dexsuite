"""Single-arm make-coffee rigid environment.

The robot must place a mug onto the tray region of a fixed coffee machine. The
target marker is attached to the machine to represent the tray goal pose.
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


MACHINE_SCALE = 0.25
MUG_SCALE = 1.0

MACHINE_Z = 0.1275
MUG_Z_OFFSET = 0.05
MACHINE_XY_BAND = 0.1
MACHINE_OFFSET_X = 0.5
MACHINE_OFFSET_Y = 0.2
MACHINE_PUSH_XY = 0.1

MUG_XY_BAND = 0.3
YAW_DEG = 45.0

MUG_TARGET_LOCAL = (-0.175, 0.0, -0.05)


SUCCESS_THR_XY = 0.06
SUCCESS_THR_YAW_RAD = 1.1


@register_env("make_coffee")
class MakeCoffeeEnv(BaseEnv):
    """Place a mug on the tray of a coffee machine."""
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
        machine_path = get_object_path("partnet/nespresso_machine")
        mug_path = get_object_path("kitchen/mug")

        self.machine = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(machine_path),
                scale=MACHINE_SCALE,
                fixed=True,
                batch_fixed_verts=True,  # per-env poses of this fixed model in batched resets
            ),
            material=gs.materials.Rigid(),
        )

        self.mug = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=str(mug_path),
                scale=MUG_SCALE,
            ),
            material=gs.materials.Rigid(),
        )

        self.mug_target = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=0.01,
                collision=False,
                fixed=True,
                pos=MUG_TARGET_LOCAL,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=(255, 0.0, 0.0)),
        )
        link_entities(
            parent_entity=self.machine,
            child_entity=self.mug_target,
            parent_link_name=self.machine.links[0].name,
            child_link_name=self.mug_target.links[0].name,
        )

        self.machine_box = self.world_aabb.half("y", "high")
        self.mug_box = self.world_aabb.half("y", "low")


    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        self._resting = 0
        self._drops = 0

        mug_pos = sample_in_aabb_center_xyz_band(
            self.mug_box,
            band_xyz=(MUG_XY_BAND, MUG_XY_BAND, 0),
            range_z=(MUG_Z_OFFSET, MUG_Z_OFFSET),
            env_idx=envs_idx,
        )

        machine_pos = sample_in_aabb_center_xyz_band(
            self.machine_box,
            range_x=(
                float(self.machine_box.max[0]) - MACHINE_XY_BAND,
                float(self.machine_box.max[0]),
            ),
            range_y=(
                float(self.machine_box.max[1]) - MACHINE_XY_BAND,
                float(self.machine_box.max[1]),
            ),
            range_z=(MACHINE_Z, MACHINE_Z),
            env_idx=envs_idx,
        )
        machine_pos = machine_pos.reshape(-1, 3)
        machine_pos[:, 0] += MACHINE_PUSH_XY
        machine_pos[:, 1] += MACHINE_PUSH_XY
        machine_pos = self.machine_box.clamp(machine_pos)
        if machine_pos.shape[0] == 1 and envs_idx is None:
            machine_pos = machine_pos.squeeze(0)

        yaw_limit_rad = math.radians(YAW_DEG)
        yaw_randomizer = YawRandomizer(yaw_range=(yaw_limit_rad / 3, yaw_limit_rad))
        machine_quat = yaw_randomizer.quat(env_idx=envs_idx,)

        self.machine.set_pos(machine_pos, envs_idx=envs_idx)
        self.machine.set_quat(machine_quat, envs_idx=envs_idx)

        self.mug.set_pos(mug_pos, envs_idx=envs_idx)
        self.mug.set_quat(yaw_randomizer.quat(env_idx=envs_idx,), envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        return {
            "machine_pos": self.machine.get_pos(),
            "machine_quat": self.machine.get_quat(),
            "mug_pos": self.mug.get_pos(),
            "mug_quat": self.mug.get_quat(),
            "mug_target_pos": self.mug_target.get_pos(),
            "mug_target_quat": self.mug_target.get_quat(),
        }

    def _is_success(self, obs) -> torch.Tensor:
        mug_position = obs["state"]["other"]["mug_pos"]
        mug_target_position = obs["state"]["other"]["mug_target_pos"]
        mug_to_target_offset = mug_position - mug_target_position
        mug_to_target_distance = torch.norm(mug_to_target_offset, dim=-1)

        within_xy = mug_to_target_distance < SUCCESS_THR_XY

        mug_yaw = quat_to_rpy_wxyz_torch(obs["state"]["other"]["mug_quat"])[..., 2]
        target_yaw = quat_to_rpy_wxyz_torch(
            obs["state"]["other"]["mug_target_quat"],
        )[..., 2]
        yaw_delta = torch.atan2(
            torch.sin(mug_yaw - target_yaw),
            torch.cos(mug_yaw - target_yaw),
        )
        yaw_misaligned_enough = torch.abs(yaw_delta) > SUCCESS_THR_YAW_RAD

        return within_xy & yaw_misaligned_enough
