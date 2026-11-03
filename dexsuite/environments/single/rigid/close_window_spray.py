"""Single-arm window environment.

A double-hung window (Artiverse asset) stands on the table facing the robot with its lower sash
raised half a metre and the latch turned. The robot pushes the sash shut by its lift rail (a bar on the
front of the sash's bottom rail, world z ~0.58 when raised), then picks the trigger spray bottle from
the side and sprays onto the closed lower pane. Spray that hits the glass is drawn into
a texture attached to the sash.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import joint_q, quat_rotate_wxyz, set_joint_dynamics, set_joint_q
from dexsuite.utils.paint_canvas import PaintCanvas
from dexsuite.utils.circular_spray import CircularSprayProgress, PaneSprayProgress, spray_axis_uv
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 900

TABLE_Z = -0.006  # vention table top (arena shifts the table so its top sits just below z = 0)
NOSLIP_ITERATIONS = 5

# Window (frame origin at mid-height; -y of the asset = front, yawed to face the robot at -x)
WINDOW_POS = (0.85, 0.0, TABLE_Z + 0.60)  # front face x 0.82; lift rail at ~0.79 is the far edge of the Franka push-down reach
WINDOW_YAW_DEG = -90.0
SASH_JOINT = "joint_2_to_4"
SASH_LINK = "link_2_panel"
LATCH_JOINT = "joint_1_to_2"
SASH_START = 0.50
LATCH_START = 1.445
SASH_MAX = 0.5356
SASH_CLOSED = 0.02

# Glass canvas on the lower sash (sash-link frame): pane x +-0.3488, z [-0.5124, -0.0047], front face y -0.0146
PANE_SIZE = (0.6976, 0.5077)
PANE_RES = (384, 512)
PANE_LOCAL_POS = (0.0, -0.016, -0.2586)
PANE_LOCAL_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)  # canvas +z -> sash -y
GLASS_RGB = (70, 95, 120)
MIST_RGB = (255, 255, 255)

# Spray bottle (upright; the asset is 25 cm tall, loaded at BOTTLE_SCALE -> 20 cm)
BOTTLE_SCALE = 0.8
BOTTLE_XY = (0.40, -0.30)
BOTTLE_BAND_XY = (0.10, 0.10)  # +-10 cm
BOTTLE_Z = TABLE_Z + 0.1255 * BOTTLE_SCALE
BOTTLE_YAW_DEG = 152.5  # nozzle toward +x (the window)
BOTTLE_YAW_JITTER_DEG = 15.0
# The URDF container is 0.9 kg (about 1 kg with the head); a full household trigger sprayer is ~0.45 kg.
BOTTLE_BODY_MASS_KG = 0.375
TRIGGER_JOINT = "joint_2_to_3"
TRIGGER_TRAVEL = 0.3927
TRIGGER_STIFFNESS = 0.2  # N m / rad return spring (~2 N at the fingertip to reach the spray gate)
# A grippy body keeps a palm wrap from rocking during carry. The trigger and index pad
# are slick: the pad must slide along the blade as the trigger swings about its
# pivot (pair friction is the max of the two geoms, so both are set).
BOTTLE_BODY_FRICTION = 3.0
TRIGGER_FRICTION = 0.3
NOZZLE_LOCAL = tuple(BOTTLE_SCALE * v for v in (-0.0358, -0.0182, 0.1182))  # container (base link) frame
SPRAY_DIR_LOCAL = (-0.887, -0.462, 0.0)

# Spray model
PULL_ON_FRAC = 0.7
CONE_HALF_ANGLE = math.radians(10.0)
RAYS_PER_STEP = 24
SPRAY_NEAR = 0.05
SPRAY_FAR = 0.5
SPLAT_RADIUS = 0.02
SPLAT_STRENGTH = 0.5

# User-requested nozzle-to-glass range; tolerate1cm of native tracking error.
PANE_SPRAY_DISTANCE_MIN = 0.15
PANE_SPRAY_DISTANCE_MAX = 0.35
PANE_SPRAY_DISTANCE_TOLERANCE = 0.01

# Historical circle diagnostics are retained for compatibility, not success.
CIRCLE_RADIUS_MIN = 0.07
CIRCLE_RADIUS_MAX = 0.14
CIRCLE_SECTORS = 24
CIRCLE_REQUIRED_SECTORS = 22


@register_env("close_window_spray")
class CloseWindowSprayEnv(BaseEnv):
    """Lower the sash, then actually spray the glass with the trigger bottle."""

    VIEWER_SHADOWS = False

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        if int(sim.n_envs) != 1:
            raise ValueError("close_window_spray paints one shared texture; it supports n_envs=1 only")
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
        self.window = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("task/window")),
                fixed=True,
                pos=WINDOW_POS,
                euler=(0.0, 0.0, WINDOW_YAW_DEG),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self.canvas = PaintCanvas(
            self.scene,
            size=PANE_SIZE,
            res=PANE_RES,
            base_rgb=GLASS_RGB,
            paint_rgb=MIST_RGB,
            pos=PANE_LOCAL_POS,
            quat=PANE_LOCAL_QUAT,
            attach_to=self.window,
            parent_link_name=SASH_LINK,
        )
        self.bottle = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("tools/spray_bottle_blue")),
                scale=BOTTLE_SCALE,
                fixed=False,
                pos=(BOTTLE_XY[0], BOTTLE_XY[1], BOTTLE_Z),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self._spray_active = False
        self.pane_spray_progress = PaneSprayProgress()
        self.circle_progress = CircularSprayProgress(
            PANE_SIZE, radius_min=CIRCLE_RADIUS_MIN, radius_max=CIRCLE_RADIUS_MAX,
            sectors=CIRCLE_SECTORS, required_sectors=CIRCLE_REQUIRED_SECTORS,
        )

    def _on_episode_start(self, envs_idx) -> None:
        set_joint_q(self.window, SASH_JOINT, SASH_START, envs_idx)
        set_joint_q(self.window, LATCH_JOINT, LATCH_START, envs_idx)
        pos = sample_around_point_xy(BOTTLE_XY, band_xy=BOTTLE_BAND_XY, z=BOTTLE_Z, env_idx=envs_idx)
        yaw0, jitter = math.radians(BOTTLE_YAW_DEG), math.radians(BOTTLE_YAW_JITTER_DEG)
        self.bottle.set_pos(pos, envs_idx=envs_idx)
        self.bottle.set_quat(YawRandomizer(yaw_range=(yaw0 - jitter, yaw0 + jitter)).quat(env_idx=envs_idx), envs_idx=envs_idx)
        set_joint_q(self.bottle, TRIGGER_JOINT, 0.0, envs_idx)
        set_joint_dynamics(self.bottle, TRIGGER_JOINT, stiffness=TRIGGER_STIFFNESS)
        self.bottle.get_link("link_1_container").set_mass(BOTTLE_BODY_MASS_KG)
        for geom in self.bottle.get_link("link_1_container").geoms:
            geom.set_friction(BOTTLE_BODY_FRICTION)
        for geom in self.bottle.get_link("link_2_handle").geoms:
            geom.set_friction(TRIGGER_FRICTION)
        hand = getattr(self.robot, "hand_ent", None)
        for link in getattr(hand, "links", ()):
            if "index" in link.name and link.name.endswith("_DP"):
                for geom in link.geoms:
                    geom.set_friction(TRIGGER_FRICTION)
        self.canvas.reset()
        self._spray_active = False
        self.pane_spray_progress.reset()
        self.circle_progress.reset()

    def _nozzle(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, quat = self.bottle.get_pos(), self.bottle.get_quat()
        local = torch.tensor(NOZZLE_LOCAL, dtype=pos.dtype, device=pos.device).expand_as(pos)
        dir_local = torch.tensor(SPRAY_DIR_LOCAL, dtype=pos.dtype, device=pos.device).expand_as(pos)
        return pos + quat_rotate_wxyz(quat, local), quat_rotate_wxyz(quat, dir_local)

    def _step(self) -> None:
        closed = float(joint_q(self.window, SASH_JOINT).reshape(-1)[0]) <= SASH_CLOSED
        if not closed:
            self.pane_spray_progress.reset()
            self.circle_progress.reset()
        pull = float(joint_q(self.bottle, TRIGGER_JOINT).reshape(-1)[0]) / TRIGGER_TRAVEL
        self._spray_active = pull >= PULL_ON_FRAC
        if not self._spray_active:
            return
        origin, direction = self._nozzle()
        uv, pts = self.canvas.spray_hits(
            origin, direction, half_angle_rad=CONE_HALF_ANGLE, n_rays=RAYS_PER_STEP, near=SPRAY_NEAR, far=SPRAY_FAR
        )
        if uv.shape[0]:
            self.canvas.splat(uv, SPLAT_RADIUS, SPLAT_STRENGTH)
            self.canvas.upload(pts)
        pane_pos, pane_rotation = self.canvas.pose()
        axis_uv = spray_axis_uv(
            origin.detach().cpu().numpy(), direction.detach().cpu().numpy(),
            pane_pos, pane_rotation, PANE_SIZE, near=SPRAY_NEAR, far=SPRAY_FAR,
        )
        self.circle_progress.update(closed=closed, active=self._spray_active, axis_uv=axis_uv, hit_uv=uv)
        nozzle_depth = float((origin.detach().cpu().numpy().reshape(3)-pane_pos) @ pane_rotation[:, 2])
        if not math.isfinite(nozzle_depth):
            raise ValueError("Nonfinite measured nozzle-to-pane distance")
        at_distance = (PANE_SPRAY_DISTANCE_MIN-PANE_SPRAY_DISTANCE_TOLERANCE <= nozzle_depth
                       <= PANE_SPRAY_DISTANCE_MAX+PANE_SPRAY_DISTANCE_TOLERANCE)
        self.pane_spray_progress.update(closed=closed, active=self._spray_active and at_distance, hit_uv=uv)

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        sash_q = joint_q(self.window, SASH_JOINT)
        trigger_q = joint_q(self.bottle, TRIGGER_JOINT)
        nozzle_pos, nozzle_dir = self._nozzle()
        pane_pos, pane_quat = self.canvas.entity.get_pos(), self.canvas.entity.get_quat()
        normal_local = torch.tensor([0.0, 0.0, 1.0], dtype=pane_pos.dtype, device=pane_pos.device).expand_as(pane_pos)
        return {
            "sash_q": sash_q,
            "sash_frac": (sash_q / SASH_MAX).clamp(0.0, 1.0),
            "sash_closed": (sash_q <= SASH_CLOSED).to(torch.float32),
            "latch_q": joint_q(self.window, LATCH_JOINT),
            "sash_pos": self.window.get_link(SASH_LINK).get_pos(),
            "bottle_pos": self.bottle.get_pos(),
            "bottle_quat": self.bottle.get_quat(),
            "bottle_vel": self.bottle.get_vel(),
            "trigger_q": trigger_q,
            "pull_frac": (trigger_q / TRIGGER_TRAVEL).clamp(0.0, 1.0),
            "nozzle_pos": nozzle_pos,
            "nozzle_dir": nozzle_dir,
            "spray_active": torch.full_like(trigger_q, float(self._spray_active)),  # decided before this step's physics
            "paint_debug_spheres": torch.full_like(trigger_q, float(self.canvas.uses_debug)),
            "pane_coverage": torch.full_like(trigger_q, self.canvas.coverage()),
            "pane_spray_steps": torch.full_like(trigger_q, float(self.pane_spray_progress.steps)),
            "pane_spray_complete": torch.full_like(trigger_q, float(self.pane_spray_progress.success)),
            "circle_sector_coverage": torch.full_like(trigger_q, self.circle_progress.coverage),
            "circle_spray_steps": torch.full_like(trigger_q, float(self.circle_progress.steps)),
            "circle_complete": torch.full_like(trigger_q, float(self.circle_progress.success)),
            "pane_center": pane_pos,
            "pane_normal": quat_rotate_wxyz(pane_quat, normal_local),
        }

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        closed = other["sash_q"][..., 0] <= SASH_CLOSED
        sprayed = other["pane_spray_complete"][..., 0] > .5
        return closed & sprayed

    def _is_failure(self, obs) -> torch.Tensor:
        return obs["state"]["other"]["bottle_pos"][..., 2] < TABLE_Z - 0.1
