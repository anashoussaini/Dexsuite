"""Single-arm safe environment.

A safe (Artiverse asset) stands in the +x/+y quadrant with its door facing -y. The door is bolted
until the lever handle has been turned a quarter turn; the robot must turn the handle, then swing
the door open.
"""

from __future__ import annotations

import math

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path
from dexsuite.utils.articulation_utils import joint_q, quat_rotate_wxyz, set_joint_dynamics, set_joint_q
from dexsuite.utils.randomizers import YawRandomizer, sample_around_point_xy

SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 600

TABLE_Z = -0.006  # vention table top (arena shifts the table so its top sits just below z = 0)
NOSLIP_ITERATIONS = 5

SAFE_XY = (0.58, 0.60)
SAFE_BAND_XY = 0.03
SAFE_YAW_JITTER_DEG = 10.0  # yaw 0 = door facing -y
SAFE_Z = TABLE_Z + 0.30  # feet on the table

DOOR_JOINT = "joint_2_to_1"
HANDLE_JOINT = "joint_3_to_2"
DIAL_JOINT = "joint_4_to_2"
DOOR_LINK = "link_2_door"
HANDLE_LINK = "link_3_handle"
DOOR_EDGE_LOCAL = (-0.272, 0.0287, 0.0)  # free edge of the door slab (mid-thickness), door-link frame
DOOR_MAX = math.pi / 2

# Bolt: the door dof carries a large Coulomb friction until the handle has turned UNLOCK_TURN
LOCKED_FRICTION = 1.0e4  # bolted: holds against a 1000 N m shove (60 N m gave way to a hard push)
UNLOCKED_FRICTION = 3.0
UNLOCK_TURN = math.pi / 2

# Success
DOOR_OPEN_MIN = math.pi / 4


@register_env("open_safe")
class OpenSafeEnv(BaseEnv):
    """Turn the handle a quarter turn, then open the safe door."""

    VIEWER_SHADOWS = False

    def __init__(self, *, robot, cameras, sim, render_mode: str | None, seed: int | None = None, **scene_kw):
        if int(sim.n_envs) != 1:
            raise ValueError("open_safe switches one shared door friction (the bolt); it supports n_envs=1 only")
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
        self.safe = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(get_object_path("task/safe")),
                fixed=True,
                pos=(SAFE_XY[0], SAFE_XY[1], SAFE_Z),
                default_armature=1e-3,
            ),
            material=gs.materials.Rigid(),
        )
        self._unlocked = False

    def _on_episode_start(self, envs_idx) -> None:
        pos = sample_around_point_xy(SAFE_XY, band_xy=SAFE_BAND_XY, z=SAFE_Z, env_idx=envs_idx)
        jitter = math.radians(SAFE_YAW_JITTER_DEG)
        quat = YawRandomizer(yaw_range=(-jitter, jitter)).quat(env_idx=envs_idx)
        self.safe.set_pos(pos, envs_idx=envs_idx)
        self.safe.set_quat(quat, envs_idx=envs_idx)
        for name in (DOOR_JOINT, HANDLE_JOINT, DIAL_JOINT):
            set_joint_q(self.safe, name, 0.0, envs_idx)
        set_joint_dynamics(self.safe, DOOR_JOINT, friction=LOCKED_FRICTION)
        self._unlocked = False

    def _step(self) -> None:
        if self._unlocked:
            return
        turn = float(joint_q(self.safe, HANDLE_JOINT).abs().max())
        if turn >= UNLOCK_TURN:
            set_joint_dynamics(self.safe, DOOR_JOINT, friction=UNLOCKED_FRICTION)
            self._unlocked = True

    def _get_extra_obs(self) -> dict[str, torch.Tensor]:
        handle_q = joint_q(self.safe, HANDLE_JOINT)
        door_q = joint_q(self.safe, DOOR_JOINT)
        door = self.safe.get_link(DOOR_LINK)
        door_pos, door_quat = door.get_pos(), door.get_quat()
        edge_local = torch.tensor(DOOR_EDGE_LOCAL, dtype=door_pos.dtype, device=door_pos.device).expand_as(door_pos)
        unlocked = torch.full_like(door_q, float(self._unlocked))
        return {
            "safe_pos": self.safe.get_pos(),
            "safe_quat": self.safe.get_quat(),
            "handle_q": handle_q,
            "handle_turn": handle_q.abs(),
            "handle_pos": self.safe.get_link(HANDLE_LINK).get_pos(),
            "door_q": door_q,
            "door_frac": (door_q / DOOR_MAX).clamp(0.0, 1.0),
            "door_pos": door_pos,
            "door_quat": door_quat,
            "door_edge_pos": door_pos + quat_rotate_wxyz(door_quat, edge_local),
            "dial_q": joint_q(self.safe, DIAL_JOINT),
            "unlocked": unlocked,
        }

    def _is_success(self, obs) -> torch.Tensor:
        other = obs["state"]["other"]
        unlocked = (other["handle_turn"][..., 0] >= UNLOCK_TURN) | (other["unlocked"][..., 0] > 0.5)
        opened = other["door_q"][..., 0] >= DOOR_OPEN_MIN
        return unlocked & opened
