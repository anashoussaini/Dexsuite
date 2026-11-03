"""Single-arm toaster environment.

A slice of toast lies flat on the table; the robot must stand it in one of the two slots of the
SMEG toaster (Artiverse asset, 1.5x) and push the lever down.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import joint_q, quat_rotate_inverse_wxyz, set_joint_q
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 600  # 30 s at 20 Hz

TABLE_Z = -0.006  # vention table top (arena shifts the table so its top sits just below z = 0)
NOSLIP_ITERATIONS = 5

# Toaster (asset frame is unscaled metres; everything below is pre-multiplied by SCALE)
SCALE = 1.5
TOASTER_XY = (0.60, 0.22)
TOASTER_BAND_XY = 0.02
TOASTER_YAW_DEG = 90.0  # slots along world y, lever end toward -y
TOASTER_YAW_JITTER_DEG = 8.0
TOASTER_Z = TABLE_Z + 0.0618 * SCALE  # body bottom on the table
SLOT_X = (-0.042 * SCALE, 0.0503 * SCALE)
SLOT_Y = (0.0072 * SCALE, 0.0286 * SCALE)  # |y| band of each slot (divider .. outer wall)
SLOT_FLOOR_Z = 0.0233 * SCALE
RIM_Z = 0.0602 * SCALE
LEVER_JOINT = "joint_5_to_1"
DIAL_JOINT = "joint_4_to_1"
LEVER_TRAVEL = 0.054 * SCALE  # lever q in [-LEVER_TRAVEL, 0], 0 = up

# Toast (Dexsuite food/toast GLB, centimetres -> metres; 10 x 9.9 x 1.2 cm slab)
TOAST_SCALE = 0.01
TOAST_THICKNESS_SCALE = 1.5
TOAST_RHO = 350.0  # kg/m^3 -> ~42 g with the thicker slice
TOAST_EULER = (0.0, 0.0, 0.0)
# Scale both the half-thickness and the mesh's off-centre origin in Z.
TOAST_SPAWN_Z = TABLE_Z + (0.004 - 0.0198) * TOAST_THICKNESS_SCALE
TOAST_XY = (0.45, -0.20)
TOAST_BAND_XY = (0.03, 0.04)

# Success
# Slot-width variants (2026-10-05, HOIR study): wider COLLISION slot walls in toaster_smeg_<variant>.urdf, the
# visual mesh is unchanged. env kwarg slot_variant (or env var DEXSUITE_TOASTER_SLOT): narrow (the asset's 3.2 cm slot,
# SLOT_Y) | wide_m (4.2 cm) | wide_b (5.4 cm). Owner 2026-10-07: wide_b is the DEFAULT ("default" = wide_b).
SLOT_Y_VARIANTS = {"narrow": SLOT_Y, "wide_m": (0.0039 * SCALE, 0.0319 * SCALE), "wide_b": (0.0010 * SCALE, 0.0370 * SCALE)}
DEFAULT_SLOT_VARIANT = "wide_b"
SLOT_X_MARGIN = 0.01
SLOT_Y_MARGIN = 0.005
MIN_DEPTH_BELOW_RIM = 0.03
UPRIGHT_MIN_HEIGHT = 0.06
REST_LIN_V_MAX = 0.05
LEVER_DOWN_FRAC = 0.8


@register_env("put_bread_toaster")
class PutBreadToasterEnv(BaseEnv):
    """Stand the toast in a toaster slot and push the lever down."""

    VIEWER_SHADOWS = False

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        self.slot_variant = str(scene_kw.pop("slot_variant", os.environ.get("DEXSUITE_TOASTER_SLOT", DEFAULT_SLOT_VARIANT)))
        if self.slot_variant == "default":
            self.slot_variant = DEFAULT_SLOT_VARIANT
        if self.slot_variant not in SLOT_Y_VARIANTS:
            raise ValueError(f"slot_variant must be one of {sorted(SLOT_Y_VARIANTS)} (or default = {DEFAULT_SLOT_VARIANT})")
        self.slot_y = SLOT_Y_VARIANTS[self.slot_variant]
        if self.slot_variant != DEFAULT_SLOT_VARIANT:
            print(f"[put_bread_toaster] slot_variant={self.slot_variant} slot |y| band {self.slot_y}", flush=True)
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

    def _toaster_file(self) -> str:
        path = Path(get_object_path("kitchen/toaster_smeg"))
        if self.slot_variant == "narrow":
            return str(path)
        variant = (path if path.is_dir() else path.parent) / f"toaster_smeg_{self.slot_variant}.urdf"
        if not variant.is_file():
            raise FileNotFoundError(variant)
        return str(variant)

    def _setup_scene(self) -> None:
        self.toaster = self.scene.add_entity(
            gs.morphs.URDF(
                file=self._toaster_file(),
                scale=SCALE,
                fixed=True,
                batch_fixed_verts=True,  # per-env poses of this fixed model in batched resets
                pos=(TOASTER_XY[0], TOASTER_XY[1], TOASTER_Z),
                euler=(0.0, 0.0, TOASTER_YAW_DEG),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self.toast = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(get_object_path("food/toast")),
                scale=(TOAST_SCALE, TOAST_SCALE, TOAST_SCALE * TOAST_THICKNESS_SCALE),
                pos=(TOAST_XY[0], TOAST_XY[1], TOAST_SPAWN_Z),
                euler=TOAST_EULER,
                merge_submeshes_for_collision=True,
            ),
            material=gs.materials.Rigid(rho=TOAST_RHO),
        )

    def _on_episode_start(self, envs_idx) -> None:
        toaster_pos = sample_around_point_xy(TOASTER_XY, band_xy=TOASTER_BAND_XY, z=TOASTER_Z, env_idx=envs_idx)
        jitter = math.radians(TOASTER_YAW_JITTER_DEG)
        yaw0 = math.radians(TOASTER_YAW_DEG)
        toaster_quat = YawRandomizer(yaw_range=(yaw0 - jitter, yaw0 + jitter)).quat(env_idx=envs_idx)
        self.toaster.set_pos(toaster_pos, envs_idx=envs_idx)
        self.toaster.set_quat(toaster_quat, envs_idx=envs_idx)
        set_joint_q(self.toaster, LEVER_JOINT, 0.0, envs_idx)
        set_joint_q(self.toaster, DIAL_JOINT, 0.0, envs_idx)

        toast_pos = sample_around_point_xy(TOAST_XY, band_xy=TOAST_BAND_XY, z=TOAST_SPAWN_Z, env_idx=envs_idx)
        base = tuple(self._toast_base_quat.tolist())
        toast_quat = YawRandomizer().quat(base_quat=base, frame="world", env_idx=envs_idx)
        self.toast.set_pos(toast_pos, envs_idx=envs_idx)
        self.toast.set_quat(toast_quat, envs_idx=envs_idx)

    @property
    def _toast_base_quat(self) -> torch.Tensor:
        r, p, y = (math.radians(a) for a in TOAST_EULER)
        cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
        return torch.tensor(
            [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
            dtype=torch.float32,
            device=self.device,
        )

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        aabb = self.toast.get_AABB()
        lo, hi = aabb[..., 0, :], aabb[..., 1, :]
        center = 0.5 * (lo + hi)
        toaster_pos = self.toaster.get_pos()
        toaster_quat = self.toaster.get_quat()
        local = quat_rotate_inverse_wxyz(toaster_quat, center - toaster_pos)
        lever_q = joint_q(self.toaster, LEVER_JOINT)
        lever_frac = (-lever_q / LEVER_TRAVEL).clamp(0.0, 1.0)

        abs_y = local[..., 1].abs()
        in_xy = (
            (local[..., 0] >= SLOT_X[0] - SLOT_X_MARGIN)
            & (local[..., 0] <= SLOT_X[1] + SLOT_X_MARGIN)
            & (abs_y >= self.slot_y[0] - SLOT_Y_MARGIN)
            & (abs_y <= self.slot_y[1] + SLOT_Y_MARGIN)
        )
        deep = lo[..., 2] <= toaster_pos[..., 2] + RIM_Z - MIN_DEPTH_BELOW_RIM
        upright = (hi[..., 2] - lo[..., 2]) >= UPRIGHT_MIN_HEIGHT
        in_slot = (in_xy & deep & upright).to(torch.float32)[..., None]
        return {
            "toast_pos": center,
            "toast_quat": self.toast.get_quat(),
            "toast_vel": self.toast.get_vel(),
            "toast_ang": self.toast.get_ang(),
            "toast_aabb_lo": lo,
            "toast_aabb_hi": hi,
            "toast_in_toaster_frame": local,
            "toaster_pos": toaster_pos,
            "toaster_quat": toaster_quat,
            "lever_q": lever_q,
            "lever_frac": lever_frac,
            "dial_q": joint_q(self.toaster, DIAL_JOINT),
            "toast_in_slot": in_slot,
        }

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        in_slot = other["toast_in_slot"][..., 0] > 0.5
        lever_down = other["lever_frac"][..., 0] >= LEVER_DOWN_FRAC
        still = torch.linalg.norm(other["toast_vel"], dim=-1) < REST_LIN_V_MAX
        return in_slot & lever_down & still

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["toast_pos"][..., 2] < TABLE_Z - 0.1
