"""Bimanual fold-glasses rigid environment.

Two arms share a workspace and must fold a pair of articulated glasses resting
on a tray.
"""

from __future__ import annotations

from typing import Any

import genesis as gs
import torch

from dexsuite.core.registry import register_env
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils import get_object_path

# Simulation
SIM_DT = 0.01
SUBSTEPS = 2
HORIZON = 40_000

# Tray
TRAY_BASE = (0.35, 0.21, 0.00)
TRAY_EULER = (0.0, 0.0, 180.0)
TRAY_SCALE = 0.0017  # OBJ in millimeters -> meters

# Task
GLASSES_SCALE = 0.10

# Success / failure thresholds
REST_FRAMES = 2
REST_LIN_V_MAX = 0.15
REST_ANG_V_MAX = 2.00
TCP_CLEARANCE = 0.06
HINGE_THR_HI = abs(1.39)  # approx 80 deg
HINGE_THR_LO = abs(1.047)  # approx 60 deg
REQUIRE_TCP_CLEAR = False  # if True, require hands-away to count as success


@register_env("bimanual_fold_glasses")
class BimanualFoldGlassesEnv(BaseEnv):
    """Two arms fold a pair of glasses that rests on a tray."""

    allowed_modes = ("bimanual",)

    def __init__(
        self,
        *,
        robot,
        cameras,
        sim,
        render_mode: str | None = None,
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
        self._resting = 0

    # -------------------------- scene setup --------------------------
    def _setup_scene(self) -> None:
        tray_obj = get_object_path("tools/tray/tray.obj")
        self.tray = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(tray_obj),
                scale=TRAY_SCALE,
                pos=TRAY_BASE,
                euler=TRAY_EULER,
                fixed=True,
                decimate_face_num=100,
                convexify=False,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Plastic(color=(0.85, 0.0, 0.85), roughness=0.6),
        )

        glass_urdf = get_object_path("partnet/glasses/mobility.urdf")
        self.glasses = self.scene.add_entity(
            morph=gs.morphs.URDF(file=str(glass_urdf), scale=GLASSES_SCALE),
            material=gs.materials.Rigid(),
        )

    # ----------------------- episode lifecycle -----------------------
    def _on_episode_start(self, envs_idx: torch.Tensor | None) -> None:
        self._resting = 0
        self._fold_latched = [False, False]

        num_reset_envs = (
            self.n_envs if envs_idx is None else int(torch.as_tensor(envs_idx).numel())
        )

        tray_pos = torch.tensor(TRAY_BASE, dtype=torch.float32, device=self.device)
        tray_quat = torch.tensor((0.7071068, 0.0, 0.0, -0.7071068), device=self.device)
        if num_reset_envs > 1 or envs_idx is not None:
            tray_pos = tray_pos.unsqueeze(0).expand(num_reset_envs, -1).contiguous()
            tray_quat = tray_quat.unsqueeze(0).expand(num_reset_envs, -1).contiguous()

        self.tray.set_pos(tray_pos, envs_idx=envs_idx)
        self.tray.set_quat(tray_quat, envs_idx=envs_idx)

        # place glasses above tray; let gravity settle them
        tray_aabb = self.tray.get_AABB()
        tray_aabb_min, tray_aabb_max = (
            (tray_aabb[0], tray_aabb[1])
            if tray_aabb.ndim == 2
            else (tray_aabb[:, 0, :], tray_aabb[:, 1, :])
        )
        top_z = tray_aabb_max[..., 2]
        tray_center_x = (tray_aabb_min[..., 0] + tray_aabb_max[..., 0]) * 0.5
        tray_center_y = (tray_aabb_min[..., 1] + tray_aabb_max[..., 1]) * 0.5

        glasses_pos = torch.stack(
            (tray_center_x, tray_center_y, top_z + 0.06),
            dim=-1,
        ).to(self.device)
        glasses_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)

        if glasses_pos.ndim == 1 and (num_reset_envs > 1 or envs_idx is not None):
            glasses_pos = (
                glasses_pos.unsqueeze(0).expand(num_reset_envs, -1).contiguous()
            )
        if glasses_quat.ndim == 1 and (num_reset_envs > 1 or envs_idx is not None):
            glasses_quat = (
                glasses_quat.unsqueeze(0).expand(num_reset_envs, -1).contiguous()
            )

        self.glasses.set_pos(glasses_pos, envs_idx=envs_idx)
        self.glasses.set_quat(glasses_quat, envs_idx=envs_idx)

        glasses_joint_positions = self.glasses.get_dofs_position()
        if int(glasses_joint_positions.shape[0]) < 2:
            raise RuntimeError("Glasses model must expose two revolute DOFs.")

    # --------------------------- helpers -----------------------------
    def _hinge_angles(self):
        """Read the two hinge DOFs directly from the entity."""
        joints_by_name = {j.name: j for j in self.glasses.joints}
        hinge_joint_0 = joints_by_name["joint_0"]
        hinge_joint_1 = joints_by_name["joint_1"]

        def _idx_local(j):
            il = getattr(j, "dof_idx_local", None)
            return int(il[0] if isinstance(il, (list, tuple)) else il)

        hinge0_dof_index = _idx_local(hinge_joint_0)
        hinge1_dof_index = _idx_local(hinge_joint_1)

        glasses_joint_positions = self.glasses.get_dofs_position()
        return (
            float(glasses_joint_positions[hinge0_dof_index].item()),
            float(glasses_joint_positions[hinge1_dof_index].item()),
        )

    # ---------------------------- I/O --------------------------------
    def _get_extra_obs(self) -> dict[str, Any]:
        try:
            aabb = self.tray.get_AABB()
            top_z = float(aabb[1, 2].item())
        except Exception:
            top_z = 0.0
        try:
            q0, q1 = self._hinge_angles()
        except Exception:
            q0 = q1 = 0.0

        return {
            "tray_top_z": torch.tensor(
                [top_z],
                dtype=torch.float32,
                device=self.device,
            ),
            "hinge_angles": torch.tensor(
                [q0, q1],
                dtype=torch.float32,
                device=self.device,
            ),
            "glasses_pos": self.glasses.get_pos()
            if hasattr(self, "glasses")
            else torch.zeros(3, device=self.device),
        }
