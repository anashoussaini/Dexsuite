from __future__ import annotations

import math
from pathlib import Path

import genesis as gs
import numpy as np
import pytest
import torch

from dexsuite.controllers.joint_position import JointPositionController
from dexsuite.controllers.joint_torque import JointTorqueController
from dexsuite.controllers.joint_velocity import JointVelocityController
from dexsuite.controllers.osc_pose import OSCPoseController
from dexsuite.controllers.osc_pose_abs import OSCPoseAbsoluteController

# Optional IK controller
try:
    from dexsuite.controllers.ik_pose import IKPoseController

    HAS_IK = True
except Exception:
    HAS_IK = False


def _zeros(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D for your controller API
    return torch.zeros((n_envs, cols), dtype=torch.float32, device=device)


def _rand(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D for your controller API
    return (torch.rand((n_envs, cols), dtype=torch.float32, device=device) * 2.0) - 1.0


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_XML = _REPO_ROOT / "dexsuite" / "models" / "manipulators" / "franka" / "assets" / "panda.xml"


@pytest.mark.parametrize("normalized", [True, False])
def test_joint_position_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)

    ctrl = JointPositionController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
    )
    ctrl.post_build()

    box = ctrl.action_space()
    assert isinstance(box.low, np.ndarray) and isinstance(box.high, np.ndarray)
    assert box.low.dtype == np.float32 and box.high.dtype == np.float32
    assert box.shape == (ctrl.dof,)

    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.all(np.isfinite(box.low)) and np.all(np.isfinite(box.high))
        assert np.all(box.low < box.high)

    ctrl.step(_zeros(n_envs, ctrl.dof, dev))
    scene.step()
    ctrl.set_home_position()
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_joint_velocity_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
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
    assert box.shape == (ctrl.dof,)
    assert box.high.dtype == np.float32 and box.low.dtype == np.float32
    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.allclose(box.low, -1.2) and np.allclose(box.high, 1.2)

    ctrl.step(_zeros(n_envs, ctrl.dof, dev))
    scene.step()
    ctrl.step(_rand(n_envs, ctrl.dof, dev))
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_joint_torque_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)

    ctrl = JointTorqueController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
        tau_max=1.0,
    )
    ctrl.post_build()

    box = ctrl.action_space()
    assert box.shape == (ctrl.dof,)
    assert box.low.dtype == np.float32 and box.high.dtype == np.float32
    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.all(np.isfinite(box.low)) and np.all(np.isfinite(box.high))
        assert np.all(box.low < box.high)

    ctrl.step(_zeros(n_envs, ctrl.dof, dev))
    scene.step()
    ctrl.step(_rand(n_envs, ctrl.dof, dev))
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_osc_pose_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)

    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
        lin_scale=0.02,
        ang_scale=0.10,
    )
    ctrl.post_build()

    box = ctrl.action_space()
    assert box.shape == (6,) and box.dtype == np.float32
    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.all(np.isfinite(box.low)) and np.all(np.isfinite(box.high))

    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_osc_pose_abs_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)

    ctrl = OSCPoseAbsoluteController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
        lin_range=(0.25, 0.25, 0.25),
        ang_range=(math.pi / 2, math.pi / 2, math.pi / 2),
    )
    ctrl.post_build()

    box = ctrl.action_space()
    assert box.shape == (6,) and box.dtype == np.float32
    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.all(np.isfinite(box.low)) and np.all(np.isfinite(box.high))

    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()


@pytest.mark.parametrize("normalized", [True, False])
def test_osc_pose_panda_mjcf_spaces_and_step(scene_robot, normalized):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    model.model_path = str(_PANDA_XML)

    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=normalized,
        clip=True,
        lin_scale=0.02,
        ang_scale=0.10,
        mink_dt=0.05,
    )
    ctrl.post_build()

    box = ctrl.action_space()
    assert box.shape == (6,) and box.dtype == np.float32
    if normalized:
        assert np.allclose(box.low, -1.0) and np.allclose(box.high, 1.0)
    else:
        assert np.all(np.isfinite(box.low)) and np.all(np.isfinite(box.high))

    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()


@pytest.mark.skipif(not HAS_IK, reason="IKPoseController unavailable")
def test_ik_pose_batched_and_single(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
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

    # 6D absolute target [x,y,z,r,p,y]
    ctrl.step(_zeros(n_envs, 6, dev))
    scene.step()
    ctrl.step(_rand(n_envs, 6, dev))
    scene.step()
