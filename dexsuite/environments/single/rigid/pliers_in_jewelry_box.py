"""Single-arm jewelry box environment.

A closed jewelry box (Artiverse asset) and a pair of pliers lie on the table. The robot must open
the lid and place the pliers inside the box.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import joint_q, quat_rotate_inverse_wxyz, set_joint_q
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 600

TABLE_Z = -0.006  # vention table top (arena shifts the table so its top sits just below z = 0)
NOSLIP_ITERATIONS = 5

# Box (25 x 15.6 x 12.7 cm; hinge on the asset's +y edge, yawed so it is on the far side from the robot)
BOX_XY = (0.58, 0.18)
BOX_BAND_XY = (0.02, 0.03)
BOX_YAW_DEG = -90.0
BOX_YAW_JITTER_DEG = 10.0
BOX_Z = TABLE_Z + 0.0635
LID_JOINT = "joint_2_to_1"
LID_MAX = math.pi / 2
CAVITY_HALF_X = 0.1094
CAVITY_HALF_Y = 0.0625
CAVITY_FLOOR_Z = -0.0479  # box-link frame
RIM_Z = 0.0459

# Pliers (one rigid body, scaled 0.85 so the 16 x 9.7 cm footprint fits the 21.9 x 12.5 cm cavity within ~10 deg of yaw;
# at full size only +-3.5 deg would fit). Either face rests flat (tilt ~4 deg, 2.4 cm tall).
PLIERS_SCALE = 0.85
PLIERS_XY = (0.42, -0.22)
PLIERS_BAND_XY = (0.03, 0.04)
PLIERS_Z = TABLE_Z + 0.0142 * PLIERS_SCALE + 0.002  # root height at rest, plus 2 mm
PLIERS_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)

# Success
CAVITY_MARGIN = 0.01
FLOOR_TOL = 0.015  # two-sided: resting on the cavity floor, not on the wall rims
RIM_TOL = 0.03
FLAT_MAX_HEIGHT = 0.04  # 2.4 cm tall lying flat; leaning on a wall is >= 6 cm
REST_LIN_V_MAX = 0.05
LID_OPEN_MIN = math.radians(60.0)


@register_env("pliers_in_jewelry_box")
class PliersInJewelryBoxEnv(BaseEnv):
    """Open the jewelry box and put the pliers inside."""

    VIEWER_SHADOWS = False

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        scene_kw.setdefault("rigid_options", gs.options.RigidOptions(noslip_iterations=NOSLIP_ITERATIONS))
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
        self.box = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("task/jewelry_box")),
                fixed=True,
                batch_fixed_verts=True,  # per-env poses of this fixed model in batched resets
                pos=(BOX_XY[0], BOX_XY[1], BOX_Z),
                euler=(0.0, 0.0, BOX_YAW_DEG),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self.pliers = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("tools/pliers")),
                scale=PLIERS_SCALE,
                fixed=False,
                pos=(PLIERS_XY[0], PLIERS_XY[1], PLIERS_Z),
                quat=PLIERS_BASE_QUAT,
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )

    def _on_episode_start(self, envs_idx) -> None:
        pos = sample_around_point_xy(BOX_XY, band_xy=BOX_BAND_XY, z=BOX_Z, env_idx=envs_idx)
        yaw0, jitter = math.radians(BOX_YAW_DEG), math.radians(BOX_YAW_JITTER_DEG)
        self.box.set_pos(pos, envs_idx=envs_idx)
        self.box.set_quat(YawRandomizer(yaw_range=(yaw0 - jitter, yaw0 + jitter)).quat(env_idx=envs_idx), envs_idx=envs_idx)
        set_joint_q(self.box, LID_JOINT, 0.0, envs_idx)

        pliers_pos = sample_around_point_xy(PLIERS_XY, band_xy=PLIERS_BAND_XY, z=PLIERS_Z, env_idx=envs_idx)
        self.pliers.set_pos(pliers_pos, envs_idx=envs_idx)
        self.pliers.set_quat(YawRandomizer().quat(base_quat=PLIERS_BASE_QUAT, frame="world", env_idx=envs_idx), envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        aabb = self.pliers.get_AABB()
        lo, hi = aabb[..., 0, :], aabb[..., 1, :]
        center = 0.5 * (lo + hi)
        box_pos, box_quat = self.box.get_pos(), self.box.get_quat()
        local = quat_rotate_inverse_wxyz(box_quat, center - box_pos)
        in_xy = (local[..., 0].abs() <= CAVITY_HALF_X - CAVITY_MARGIN) & (local[..., 1].abs() <= CAVITY_HALF_Y - CAVITY_MARGIN)
        on_floor = (lo[..., 2] - (box_pos[..., 2] + CAVITY_FLOOR_Z)).abs() <= FLOOR_TOL
        below_rim = hi[..., 2] <= box_pos[..., 2] + RIM_Z + RIM_TOL
        flat = (hi[..., 2] - lo[..., 2]) <= FLAT_MAX_HEIGHT
        lid_q = joint_q(self.box, LID_JOINT)
        return {
            "pliers_pos": center,
            "pliers_quat": self.pliers.get_quat(),
            "pliers_vel": self.pliers.get_vel(),
            "pliers_ang": self.pliers.get_ang(),
            "pliers_aabb_lo": lo,
            "pliers_aabb_hi": hi,
            "pliers_in_box_frame": local,
            "pliers_inside": (in_xy & on_floor & below_rim & flat).to(torch.float32)[..., None],
            "box_pos": box_pos,
            "box_quat": box_quat,
            "lid_q": lid_q,
            "lid_frac": (lid_q / LID_MAX).clamp(0.0, 1.0),
        }

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        inside = other["pliers_inside"][..., 0] > 0.5
        lid_open = other["lid_q"][..., 0] >= LID_OPEN_MIN
        still = torch.linalg.norm(other["pliers_vel"], dim=-1) < REST_LIN_V_MAX
        return inside & lid_open & still

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["pliers_pos"][..., 2] < TABLE_Z - 0.1
