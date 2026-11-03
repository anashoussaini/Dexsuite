"""Single-arm graffiti environment.

A sheet of white paper lies on the table next to an aerosol can (Artiverse asset). Every episode the
same red stencil outline (``STENCIL_SHAPE``, ``STENCIL_SIZE`` wide, centred on the sheet) is drawn on the
paper as the guide of what to spray; the sheet and the can still move. Pressing the can's head sprays black paint in a
cone; paint that lands on the paper is drawn into its texture over the outline. Success is at least
``STENCIL_SUCCESS`` of the stencil interior painted with at most ``OUTSIDE_MAX`` of the paint mass
outside it.
"""

from __future__ import annotations

import math

import genesis as gs
import numpy as np
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import joint_q, quat_rotate_wxyz, set_joint_dynamics, set_joint_q
from dexsuite.utils.paint_canvas import PaintCanvas
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 600

TABLE_Z = -0.006  # vention table top (arena shifts the table so its top sits just below z = 0)
NOSLIP_ITERATIONS = 5

# Paper (A3, flat on the table)
PAPER_SIZE = (0.42, 0.297)
PAPER_RES = (384, 512)  # rows (y), cols (x)
PAPER_XY = (0.55, -0.02)
PAPER_BAND_XY = (0.02, 0.03)
PAPER_Z = TABLE_Z + 0.001
PAPER_RGB = (245, 245, 242)
PAINT_RGB = (20, 20, 24)

# Can (upright, o 9.6 x 23.7 cm incl. head)
CAN_SCALE = 0.7  # asset is 9.6 cm wide x 19.7 cm tall; 0.7 -> 6.7 x 13.8 cm, a one-hand power grasp
CAN_XY = (0.42, -0.32)
CAN_BAND_XY = (0.02, 0.03)
CAN_YAW_DEG = 90.0  # nozzle (can -y) toward world +x
CAN_YAW_JITTER_DEG = 15.0
CAN_Z = TABLE_Z + 0.1185 * CAN_SCALE
HEAD_JOINT = "joint_2_to_1"
HEAD_LINK = "link_2_head"
HEAD_TRAVEL = 0.009 * CAN_SCALE  # Genesis scales the prismatic limits with the asset
HEAD_STIFFNESS = 200.0  # N/m return spring toward the released position
NOZZLE_LOCAL = tuple(CAN_SCALE * v for v in (0.0, -0.0117, 0.104))  # head-link frame; head meshes are 1.3x wider (-0.009 * 1.3)
NOZZLE_DIR_LOCAL = (0.0, -1.0, 0.0)

# Spray model
PRESS_ON_FRAC = 0.7
CONE_HALF_ANGLE = math.radians(4.0)
RAYS_PER_STEP = 24
SPRAY_NEAR = 0.03
SPRAY_FAR = 0.45
SPLAT_RADIUS = 0.008
SPLAT_STRENGTH = 0.5

# Stencil (red guide drawn into the paper texture; ``size`` is the shape's bounding height/width)
STENCIL_SHAPES = ("circle", "square", "triangle", "letter_f")
STENCIL_RGB = (220, 30, 30)
STENCIL_LINE_M = 0.004
STENCIL_SIZE_RANGE = (0.12, 0.18)  # stencil_pose() only; the env uses the fixed stencil below
STENCIL_SHAPE = "circle"  # one fixed stencil every episode (was a random shape/size/pose until 2026-09-26)
STENCIL_SIZE = 0.15
STENCIL_CENTER = (0.0, 0.0)  # sheet-local metres
STENCIL_MARGIN = 0.03  # shape bounding box stays this far from the sheet edges
F_WIDTH = 0.7  # letter F bounding width / size
F_STROKE = 0.3  # stroke thickness / size (thin strokes made the outside-paint guard much stricter for the F)

# Success
STENCIL_SUCCESS = 0.6  # fraction of the stencil interior painted
OUTSIDE_MAX = 0.35  # fraction of the paint mass allowed outside the stencil


def stencil_half_extents(shape: str, size: float) -> tuple[float, float]:
    """Half-widths (x, y) of the shape's bounding box in metres."""
    if shape == "triangle":
        return 0.5 * size, 0.25 * math.sqrt(3.0) * size
    if shape == "letter_f":
        return 0.5 * F_WIDTH * size, 0.5 * size
    return 0.5 * size, 0.5 * size


def stencil_pose(shape: str, u: torch.Tensor) -> tuple[tuple[float, float], float]:
    """Map three uniform draws to ``((cx, cy), size)`` in sheet-local metres, keeping ``STENCIL_MARGIN`` to the edges."""
    u0, u1, u2 = (float(v) for v in u.reshape(3))
    size = STENCIL_SIZE_RANGE[0] + (STENCIL_SIZE_RANGE[1] - STENCIL_SIZE_RANGE[0]) * u0
    hx, hy = stencil_half_extents(shape, size)
    cx = (2.0 * u1 - 1.0) * max(0.5 * PAPER_SIZE[0] - STENCIL_MARGIN - hx, 0.0)
    cy = (2.0 * u2 - 1.0) * max(0.5 * PAPER_SIZE[1] - STENCIL_MARGIN - hy, 0.0)
    return (cx, cy), size


def stencil_mask(shape: str, center_xy, size: float) -> np.ndarray:
    """Boolean (H, W) interior of ``shape`` on the paper texture, ``center_xy`` in metres from the sheet centre."""
    H, W = PAPER_RES
    x = (np.arange(W) + 0.5) / W * PAPER_SIZE[0] - 0.5 * PAPER_SIZE[0] - float(center_xy[0])
    y = 0.5 * PAPER_SIZE[1] - (np.arange(H) + 0.5) / H * PAPER_SIZE[1] - float(center_xy[1])  # row 0 = top (+y)
    X, Y = np.meshgrid(x, y)
    h = 0.5 * size
    if shape == "circle":
        return X**2 + Y**2 <= h**2
    if shape == "square":
        return (np.abs(X) <= h) & (np.abs(Y) <= h)
    if shape == "triangle":  # equilateral, side = size, apex up
        hy = 0.25 * math.sqrt(3.0) * size
        return (Y >= -hy) & (np.abs(X) <= (hy - Y) / math.sqrt(3.0))
    if shape == "letter_f":
        hw, t = 0.5 * F_WIDTH * size, F_STROKE * size
        stem = (X >= -hw) & (X <= -hw + t) & (np.abs(Y) <= h)
        top = (np.abs(X) <= hw) & (Y >= h - t) & (Y <= h)
        mid = (X >= -hw) & (X <= hw - 0.25 * F_WIDTH * size) & (np.abs(Y) <= 0.5 * t)
        return stem | top | mid
    raise ValueError(f"unknown stencil shape {shape!r}")


@register_env("spray_graffiti")
class SprayGraffitiEnv(BaseEnv):
    """Spray paint from the aerosol can onto the paper."""

    VIEWER_SHADOWS = False

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        if int(sim.n_envs) != 1:
            raise ValueError("spray_graffiti paints one shared texture; it supports n_envs=1 only")
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
        self.canvas = PaintCanvas(
            self.scene,
            size=PAPER_SIZE,
            res=PAPER_RES,
            base_rgb=PAPER_RGB,
            paint_rgb=PAINT_RGB,
            pos=(PAPER_XY[0], PAPER_XY[1], PAPER_Z),
        )
        self.can = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("tools/aerosol_can")),
                scale=CAN_SCALE,
                fixed=False,
                pos=(CAN_XY[0], CAN_XY[1], CAN_Z),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self._spray_active = False
        self._last_hit_xy = torch.zeros(2)
        self._stencil_id = 0
        self._stencil_center = torch.zeros(2)
        self._stencil_size = 0.0

    def _on_episode_start(self, envs_idx) -> None:
        paper_pos = sample_around_point_xy(PAPER_XY, band_xy=PAPER_BAND_XY, z=PAPER_Z, env_idx=envs_idx)
        self.canvas.entity.set_pos(paper_pos, envs_idx=envs_idx)
        can_pos = sample_around_point_xy(CAN_XY, band_xy=CAN_BAND_XY, z=CAN_Z, env_idx=envs_idx)
        self.can.set_pos(can_pos, envs_idx=envs_idx)
        yaw0, jitter = math.radians(CAN_YAW_DEG), math.radians(CAN_YAW_JITTER_DEG)
        self.can.set_quat(YawRandomizer(yaw_range=(yaw0 - jitter, yaw0 + jitter)).quat(env_idx=envs_idx), envs_idx=envs_idx)
        set_joint_q(self.can, HEAD_JOINT, 0.0, envs_idx)
        set_joint_dynamics(self.can, HEAD_JOINT, stiffness=HEAD_STIFFNESS)
        self._stencil_id = STENCIL_SHAPES.index(STENCIL_SHAPE)
        center, size = STENCIL_CENTER, STENCIL_SIZE
        self._stencil_center = torch.tensor(center, dtype=torch.float32)
        self._stencil_size = size
        self.canvas.set_stencil(
            stencil_mask(STENCIL_SHAPES[self._stencil_id], center, size), outline_rgb=STENCIL_RGB, outline_m=STENCIL_LINE_M
        )
        self.canvas.reset()
        self._spray_active = False
        self._last_hit_xy = torch.zeros(2)

    def _nozzle(self) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.can.get_link(HEAD_LINK)
        pos, quat = head.get_pos(), head.get_quat()
        local = torch.tensor(NOZZLE_LOCAL, dtype=pos.dtype, device=pos.device).expand_as(pos)
        dir_local = torch.tensor(NOZZLE_DIR_LOCAL, dtype=pos.dtype, device=pos.device).expand_as(pos)
        return pos + quat_rotate_wxyz(quat, local), quat_rotate_wxyz(quat, dir_local)

    def _step(self) -> None:
        press = float(joint_q(self.can, HEAD_JOINT).reshape(-1)[0]) / HEAD_TRAVEL
        self._spray_active = press >= PRESS_ON_FRAC
        if not self._spray_active:
            return
        origin, direction = self._nozzle()
        uv, pts = self.canvas.spray_hits(
            origin, direction, half_angle_rad=CONE_HALF_ANGLE, n_rays=RAYS_PER_STEP, near=SPRAY_NEAR, far=SPRAY_FAR
        )
        if uv.shape[0]:
            self.canvas.splat(uv, SPLAT_RADIUS, SPLAT_STRENGTH)
            self.canvas.upload(pts)
            self._last_hit_xy = torch.as_tensor(pts[:, :2].mean(axis=0), dtype=torch.float32)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        head_q = joint_q(self.can, HEAD_JOINT)
        nozzle_pos, nozzle_dir = self._nozzle()
        coverage = torch.full_like(head_q, self.canvas.coverage())
        paper_pos = self.canvas.entity.get_pos()
        return {
            "can_pos": self.can.get_pos(),
            "can_quat": self.can.get_quat(),
            "can_vel": self.can.get_vel(),
            "head_q": head_q,
            "press_frac": (head_q / HEAD_TRAVEL).clamp(0.0, 1.0),
            "nozzle_pos": nozzle_pos,
            "nozzle_dir": nozzle_dir,
            "spray_active": torch.full_like(head_q, float(self._spray_active)),  # decided before this step's physics
            "paint_debug_spheres": torch.full_like(head_q, float(self.canvas.uses_debug)),
            "spray_hit_xy": self._last_hit_xy.to(head_q.device).expand(*head_q.shape[:-1], 2),
            "paint_coverage": coverage,
            "paper_pos": paper_pos,
            "paper_quat": self.canvas.entity.get_quat(),
            "stencil_id": torch.full_like(head_q, float(self._stencil_id)),
            "stencil_center_xy": paper_pos[..., :2] + self._stencil_center.to(paper_pos.device),
            "stencil_size": torch.full_like(head_q, self._stencil_size),
            "stencil_coverage": torch.full_like(head_q, self.canvas.coverage(region="inside")),
            "paint_outside_frac": torch.full_like(head_q, self.canvas.coverage(region="outside")),
        }

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        filled = other["stencil_coverage"][..., 0] >= STENCIL_SUCCESS
        clean = other["paint_outside_frac"][..., 0] <= OUTSIDE_MAX
        return filled & clean

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["can_pos"][..., 2] < TABLE_Z - 0.1
