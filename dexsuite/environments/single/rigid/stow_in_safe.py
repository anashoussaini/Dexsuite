"""Single-arm safe stowing environment.

The safe (Artiverse asset) stands in the +x/+y quadrant with its door closed and bolted, exactly as in
``open_safe``. A red mug sits on the table; the robot must turn the handle a quarter turn to retract the
bolt, swing the door open, put the mug inside the safe and push the door shut.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.single.rigid.open_safe import TABLE_Z, OpenSafeEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import quat_rotate_inverse_wxyz
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

HORIZON = 1200  # unbolt, open, stow, close

# Mug (YCB red mug, AABB 11.9 x 9.4 x 8.3 cm; body origin 2.83 cm above the base, handle 6.9 cm out along +x)
MUG_XY = (0.34, 0.07)  # collision hull >= 2 cm outside the door's sweep at the worst safe/mug jitter corner, 22 cm from the hand's home tcp
MUG_BAND_XY = (0.02, 0.03)
MUG_Z = TABLE_Z + 0.0283
MUG_YAW_JITTER_DEG = 15.0

# Safe cavity, safe-link frame (walls x +-0.146, back y 0.165, floor z -0.181, ceiling z 0.279; front opening y -0.207).
# Two fixed shelves split it into 14.9 / 11.5 / 17.6 cm tall compartments; the 8.3 cm mug fits any of them.
CAVITY_HALF_X = 0.146 - 0.015
DOOR_BACK_Y = -0.154  # back face of the closed door slab
CAVITY_Y = (DOOR_BACK_Y - 0.005, 0.15)  # front bound on the mug's front-most collision vertex (5 mm penetration slack), back bound on its centre
CAVITY_Z = (-0.20, 0.28)

# Success
DOOR_CLOSED = math.radians(5.0)
REST_LIN_V_MAX = 0.05


@register_env("stow_in_safe")
class StowInSafeEnv(OpenSafeEnv):
    """Unbolt and open the safe, put the mug inside, then close the door."""

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        super().__init__(robot=robot, cameras=cameras, sim=sim, render_mode=render_mode, seed=seed, **scene_kw)
        self.horizon = HORIZON

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.mug = self.scene.add_entity(
            gs.morphs.MJCF(file=str(get_object_path("kitchen/mug")), pos=(MUG_XY[0], MUG_XY[1], MUG_Z)),
            material=gs.materials.Rigid(),
        )

    def _on_episode_start(self, envs_idx) -> None:
        super()._on_episode_start(envs_idx)
        jitter = math.radians(MUG_YAW_JITTER_DEG)
        self.mug.set_pos(sample_around_point_xy(MUG_XY, band_xy=MUG_BAND_XY, z=MUG_Z, env_idx=envs_idx), envs_idx=envs_idx)
        self.mug.set_quat(YawRandomizer(yaw_range=(-jitter, jitter)).quat(env_idx=envs_idx), envs_idx=envs_idx)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        extras = super()._get_extra_obs()
        aabb = self.mug.get_AABB()
        lo, hi = aabb[..., 0, :], aabb[..., 1, :]
        safe_pos, safe_quat = extras["safe_pos"], extras["safe_quat"]
        local = quat_rotate_inverse_wxyz(safe_quat, 0.5 * (lo + hi) - safe_pos)
        verts = quat_rotate_inverse_wxyz(safe_quat[..., None, :], self.mug.get_verts() - safe_pos[..., None, :])
        inside = (
            (local[..., 0].abs() <= CAVITY_HALF_X)
            & (verts[..., 1].amin(dim=-1) >= CAVITY_Y[0])
            & (local[..., 1] <= CAVITY_Y[1])
            & (local[..., 2] >= CAVITY_Z[0])
            & (local[..., 2] <= CAVITY_Z[1])
        )
        extras.update(
            {
                "mug_pos": self.mug.get_pos(),
                "mug_quat": self.mug.get_quat(),
                "mug_vel": self.mug.get_vel(),
                "mug_aabb_lo": lo,
                "mug_aabb_hi": hi,
                "mug_in_safe_frame": local,
                "mug_inside": inside.to(torch.float32)[..., None],
            }
        )
        return extras

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        unlocked = other["unlocked"][..., 0] > 0.5
        inside = other["mug_inside"][..., 0] > 0.5
        still = torch.linalg.norm(other["mug_vel"], dim=-1) < REST_LIN_V_MAX
        closed = other["door_q"][..., 0] <= DOOR_CLOSED
        return unlocked & inside & still & closed

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["mug_pos"][..., 2] < TABLE_Z - 0.1
