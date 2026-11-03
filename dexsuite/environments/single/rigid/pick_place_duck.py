"""Single-arm rubber duck pick-and-place rigid environment.

A 9 cm, 40 g rubber duck (task/duck, built by scripts/assets/make_duck.py) stands upright on the table with a uniform
random heading; it and a bowl (kitchen/bowl at 1.7x) are each placed anywhere in the spawn box with at least 20 cm of
clear table between the duck and the bowl's rim (centres 38-60 cm apart), the bowl on any side. The robot must put the duck in the bowl.
The intended grasp is a three-finger pinch on the duck's head from above: thumb on one cheek, index and middle on the
other, palm clear of the body (HEAD_LOCAL is the head centre in the duck frame; obs other.duck_head_pos is it in
the world). Success: the duck rests inside the bowl (COM within BOWL_INNER_R - XY_MARGIN of the bowl axis, lowest
point within FLOOR_TOL of the cavity floor), still and not touching the hand, for REST_FRAMES consecutive control
ticks; any orientation counts. Failure: the duck falls off the table.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 500  # 25 s at 20 Hz (as pick_place_pan)
NOSLIP_ITERATIONS = 5
TABLE_Z = -0.006  # vention table top

# Bowl: kitchen/bowl at 1.7 (as pick_place_fruit), placed per episode (SPAWN_*). bowl.xml body sits 0.05 above the entity origin (scaled);
# collision hulls span z -0.0395 .. +0.0385 around the body, cavity floor (collision) at -0.02055 on the axis,
# smallest collision inner radius at the rim 0.0613 (all native units).
BOWL_PATH = "kitchen/bowl"
BOWL_SCALE = 1.7
BOWL_Z = TABLE_Z + (0.0395 - 0.05) * BOWL_SCALE  # entity z: collision bottom on the table (-0.02385)
BOWL_RIM_DZ = (0.05 + 0.03854) * BOWL_SCALE  # entity z -> rim top (0.1505)
BOWL_FLOOR_DZ = (0.05 - 0.02055) * BOWL_SCALE  # entity z -> collision floor on the axis (0.0501)
BOWL_INNER_R = 0.0613 * BOWL_SCALE  # 0.1042: narrowest collision opening (radius)
BOWL_OUTER_R = 0.0781 * BOWL_SCALE  # 0.1327: collision outer radius (spawn-gap bookkeeping only)
BOWL_XY = (0.50, 0.20)  # the build-time placement only; episodes sample it (SPAWN_* below)
BOWL_BAND_XY = (0.03, 0.03)

# Duck: task/duck (duck.xml). Frame x = beak, y = left, z = up, origin = COM. 9.0 x 6.9 x 7.55 cm, 40 g.
DUCK_PATH = "task/duck"
DUCK_COM_HEIGHT = 0.0311  # COM above the lowest collision point when upright
DUCK_XY_RADIUS = 0.0477  # farthest collision point from the COM in xy (tail tip)
DUCK_Z = TABLE_Z + DUCK_COM_HEIGHT + 0.002  # 2 mm drop at spawn
DUCK_XY = (0.45, -0.21)
DUCK_BAND_XY = (0.04, 0.04)
DUCK_YAW_RANGE_DEG = 180.0  # uniform heading

# Spawn: the duck and the bowl each anywhere in SPAWN_X x SPAWN_Y (centres; inside the floating
# hand's workspace and the front camera's view), the bowl SEP_MIN..SEP_MAX from the duck, so the bowl can be on any side
# of the duck. SEP_MIN: at least GAP_MIN of clear table between the duck and the bowl's rim, edge to edge
# (20 cm) = bowl outer radius + duck radius + GAP_MIN. Rejection sampling with SPAWN_TRIES candidates
# per env from the env's torch RNG (env.reset(seed) stays repeatable).
SPAWN_X = (0.36, 0.66)
SPAWN_Y = (-0.36, 0.36)
GAP_MIN = 0.20
SEP_MIN = BOWL_OUTER_R + DUCK_XY_RADIUS + GAP_MIN  # 0.3804 between centres
SEP_MAX = 0.60
SPAWN_TRIES = 256
HEAD_LOCAL = (0.010, 0.0, 0.0214)  # head centre (widest lateral section), duck frame

# Success
XY_MARGIN = 0.03  # COM within BOWL_INNER_R - XY_MARGIN (0.074 m) of the bowl axis
FLOOR_TOL = 0.04  # duck's lowest point within 4 cm of the collision floor height (>= 6 cm below the rim)
REST_LIN_V_MAX = 0.05
REST_ANG_V_MAX = 0.5
REST_FRAMES = 5


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v (..., 3) by wxyz quaternions q (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


@register_env("pick_place_duck")
class PickPlaceDuckEnv(BaseEnv):
    """Pick the rubber duck by the head and put it in the bowl."""

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
        self.bowl = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(get_object_path(BOWL_PATH)),
                scale=BOWL_SCALE,
                pos=(BOWL_XY[0], BOWL_XY[1], BOWL_Z),
                decimate=False,
                batch_fixed_verts=True,  # per-env pose of a fixed body when n_envs > 1
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Plastic(color=(0.95, 0.95, 0.95), roughness=0.35, metallic=0.0, ior=1.47),
        )
        self.duck = self.scene.add_entity(
            gs.morphs.MJCF(file=str(get_object_path(DUCK_PATH)), pos=(DUCK_XY[0], DUCK_XY[1], DUCK_Z)),
            material=gs.materials.Rigid(),
        )
        self._rest_ticks = torch.zeros(self.n_envs, dtype=torch.int64, device=self.device)
        self._head_local = torch.tensor(HEAD_LOCAL, dtype=torch.float32, device=self.device)

    def _sample_layout(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, 2) duck and bowl centres: (duck, bowl) PAIRS uniform in the spawn box, the first pair SEP_MIN..SEP_MAX
        apart (sampled jointly: with a 20 cm gap some duck positions, e.g. mid-box, have no valid bowl position)."""
        dev = self.device
        lo = torch.tensor([SPAWN_X[0], SPAWN_Y[0]], device=dev)
        span = torch.tensor([SPAWN_X[1] - SPAWN_X[0], SPAWN_Y[1] - SPAWN_Y[0]], device=dev)
        duck_c = lo + torch.rand(n, SPAWN_TRIES, 2, device=dev) * span
        bowl_c = lo + torch.rand(n, SPAWN_TRIES, 2, device=dev) * span
        d = torch.linalg.norm(bowl_c - duck_c, dim=-1)
        ok = (d >= SEP_MIN) & (d <= SEP_MAX)                           # ~22 % of pairs: P(none of 256) ~ 1e-27
        first = torch.argmax(ok.to(torch.int64), dim=1)                 # the first valid pair per env
        rows = torch.arange(n, device=dev)
        duck, bowl = duck_c[rows, first], bowl_c[rows, first]
        if not bool(ok.any(dim=1).all()):                               # practically never: the build-time layout
            bad = ~ok.any(dim=1)
            duck[bad] = torch.tensor(DUCK_XY, device=dev)
            bowl[bad] = torch.tensor(BOWL_XY, device=dev)
        return duck, bowl

    def _on_episode_start(self, envs_idx) -> None:
        n = self.n_envs if envs_idx is None else len(torch.as_tensor(envs_idx).reshape(-1))
        duck_xy, bowl_xy = self._sample_layout(n)
        z = lambda xy, h: torch.cat([xy, torch.full((xy.shape[0], 1), h, device=xy.device)], dim=-1)
        bowl_pos, duck_pos = z(bowl_xy, BOWL_Z), z(duck_xy, DUCK_Z)
        if self.n_envs == 1 and envs_idx is None:
            bowl_pos, duck_pos = bowl_pos[0], duck_pos[0]
        self.bowl.set_pos(bowl_pos, envs_idx=envs_idx)
        self.duck.set_pos(duck_pos, envs_idx=envs_idx, zero_velocity=True)
        y = math.radians(DUCK_YAW_RANGE_DEG)
        self.duck.set_quat(YawRandomizer(yaw_range=(-y, y)).quat(env_idx=envs_idx), envs_idx=envs_idx)
        if envs_idx is None:
            self._rest_ticks.zero_()
        else:
            self._rest_ticks[torch.as_tensor(envs_idx, device=self.device).long()] = 0

    def _duck_hand_contact(self) -> torch.Tensor:
        c = self.duck.get_contacts(with_entity=self.robot.hand_ent)
        if self.n_envs == 1:
            return torch.tensor([float(c["geom_a"].numel() > 0)], device=self.device)
        return c["valid_mask"].any(dim=-1, keepdim=True).to(torch.float32)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        d = self.duck.get_AABB()
        lo, hi = d[..., 0, :], d[..., 1, :]
        pos, quat = self.duck.get_pos(), self.duck.get_quat()
        vel, ang = self.duck.get_vel(), self.duck.get_ang()
        bpos = self.bowl.get_pos()
        one = torch.ones_like(bpos[..., :1])
        rim_z, floor_z, inner_r = bpos[..., 2:3] + BOWL_RIM_DZ, bpos[..., 2:3] + BOWL_FLOOR_DZ, one * BOWL_INNER_R
        r_ok = inner_r - XY_MARGIN
        touch = self._duck_hand_contact().reshape(one.shape)
        in_bowl = (torch.linalg.norm(pos[..., :2] - bpos[..., :2], dim=-1) <= r_ok[..., 0]) & (lo[..., 2] <= floor_z[..., 0] + FLOOR_TOL)
        still = (torch.linalg.norm(vel, dim=-1) < REST_LIN_V_MAX) & (torch.linalg.norm(ang, dim=-1) < REST_ANG_V_MAX)
        resting = (in_bowl & still & (touch[..., 0] < 0.5)).reshape(-1)
        self._rest_ticks = torch.where(resting, self._rest_ticks + 1, torch.zeros_like(self._rest_ticks))
        return {
            "duck_pos": pos,
            "duck_quat": quat,
            "duck_vel": vel,
            "duck_ang": ang,
            "duck_aabb_lo": lo,
            "duck_aabb_hi": hi,
            "duck_head_pos": pos + _quat_apply(quat, self._head_local.expand_as(pos)),
            "bowl_pos": bpos,
            "bowl_rim_z": rim_z,
            "bowl_floor_z": floor_z,
            "bowl_inner_radius": inner_r,
            "bowl_inner_lo": bpos[..., :2] - r_ok,
            "bowl_inner_hi": bpos[..., :2] + r_ok,
            "duck_in_bowl": in_bowl.to(torch.float32).reshape(one.shape),
            "duck_hand_contact": touch,
            "duck_rest_ticks": self._rest_ticks.to(torch.float32).reshape(one.shape),
        }

    def _is_success(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["duck_rest_ticks"][..., 0] >= REST_FRAMES

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["duck_pos"][..., 2] < TABLE_Z - 0.1
