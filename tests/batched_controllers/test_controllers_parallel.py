from __future__ import annotations

import genesis as gs
import numpy as np
import pytest
import torch

from dexsuite.controllers.joint_position import JointPositionController
from dexsuite.controllers.joint_torque import JointTorqueController
from dexsuite.controllers.joint_velocity import JointVelocityController
from dexsuite.controllers.osc_pose import OSCPoseController
from dexsuite.controllers.osc_pose_abs import OSCPoseAbsoluteController

try:
    from dexsuite.controllers.ik_pose import IKPoseController  # noqa: F401

    HAS_IK = True
except Exception:
    HAS_IK = False


def _zeros(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((n_envs, cols), dtype=torch.float32, device=device)


def _rand(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.rand((n_envs, cols), dtype=torch.float32, device=device) * 2 - 1


@pytest.mark.parametrize("normalized", [True, False])
def test_joint_position_spaces_and_step(scene_robot, normalized):
    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointPositionController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
    )
    ctrl.post_build()
    box = ctrl.action_space()
    assert isinstance(box.low, np.ndarray)
    assert box.low.dtype == np.float32
    act = _zeros(n_envs, ctrl.dof, dev)
    ctrl.step(act)
    scene.step()
    ctrl.set_home_position()
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_joint_velocity_spaces_and_step(scene_robot, normalized):
    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointVelocityController(
        entity=robot,
        model=model,
        normalized=normalized,
        v_max=1.2,
        clip=True,
    )
    ctrl.post_build()
    box = ctrl.action_space()
    assert isinstance(box.high, np.ndarray)
    assert box.high.dtype == np.float32
    ctrl.step(_zeros(n_envs, ctrl.dof, dev))
    scene.step()
    ctrl.step(_rand(n_envs, ctrl.dof, dev))
    scene.step()


def test_joint_torque_nominal_spaces_and_step(scene_robot):
    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointTorqueController(
        entity=robot,
        model=model,
        normalized=False,
        clip=True,
        tau_max=1.0,
    )
    ctrl.post_build()
    box = ctrl.action_space()
    assert box.low.dtype == np.float32
    ctrl.step(_zeros(n_envs, ctrl.dof, dev))
    scene.step()
    small = torch.full((n_envs, ctrl.dof), 0.2, dtype=torch.float32, device=dev)
    ctrl.step(small)
    scene.step()


def test_osc_pose_normalized_spaces_and_step(scene_robot):
    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        lin_scale=0.02,
        ang_scale=0.10,
    )
    ctrl.post_build()
    box = ctrl.action_space()
    assert box.shape == (6,)
    assert box.dtype == np.float32
    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()


def test_osc_pose_abs_normalized_spaces_and_step(scene_robot):
    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = OSCPoseAbsoluteController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        lin_range=(0.25, 0.25, 0.25),
        ang_range=(np.pi / 2, np.pi / 2, np.pi / 2),
    )
    ctrl.post_build()
    box = ctrl.action_space()
    assert box.shape == (6,)
    assert box.dtype == np.float32
    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()


@pytest.mark.skipif(not HAS_IK, reason="IKPoseController not available/batched yet")
def test_ik_pose_batched_step(scene_robot):
    from dexsuite.controllers.ik_pose import IKPoseController

    scene, robot, ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = IKPoseController(
        entity=robot,
        model=model,
        ik_max_iters=80,
        ik_damping=0.05,
        ik_max_step=0.2,
        ik_init="home",
    )
    ctrl.post_build()
    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()
