# ruff: noqa: S101

from __future__ import annotations

import genesis as gs
import numpy as np
import pytest
import torch

from dexsuite.controllers.joint_position import JointPositionController
from dexsuite.controllers.joint_torque import JointTorqueController
from dexsuite.controllers.joint_velocity import JointVelocityController
from dexsuite.controllers.osc_pose_abs import OSCPoseAbsoluteController

try:
    from dexsuite.controllers.ik_pose import IKPoseController  # noqa: F401

    HAS_IK = True
except Exception:
    HAS_IK = False


def _zeros(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D for this controller API
    return torch.zeros((n_envs, cols), dtype=torch.float32, device=device)


def _rand(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D for this controller API
    return (torch.rand((n_envs, cols), dtype=torch.float32, device=device) * 2.0) - 1.0


def _per_joint_force_caps(robot, dofs_idx=None):
    lo, hi = robot.get_dofs_force_range(dofs_idx_local=dofs_idx)
    if lo.ndim == 2:
        lo = lo[0]
        hi = hi[0]
    lo = lo.detach().cpu()
    hi = hi.detach().cpu()
    return torch.maximum(lo.abs(), hi.abs())


def _eef_pos_2d(robot, link_name: str) -> torch.Tensor:
    pos = robot.get_link(link_name).get_pos()
    return pos.unsqueeze(0) if pos.ndim == 1 else pos


@pytest.mark.parametrize("n_bad", [1, 2, 7])  # different mismatches
def test_joint_position_bad_shape_raises(scene_robot, n_bad):
    _scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointPositionController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
    )
    ctrl.post_build()

    bad_cols = ctrl.dof + n_bad
    bad_vec = torch.zeros((n_envs, bad_cols), dtype=torch.float32, device=dev)

    with pytest.raises(ValueError):
        ctrl.step(bad_vec)


def test_noncontiguous_broadcast_inputs_ok(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointVelocityController(
        entity=robot,
        model=model,
        normalized=True,
        v_max=1.0,
        clip=True,
    )
    ctrl.post_build()

    if n_envs == 1:
        pytest.skip("broadcast test only meaningful for batched scenes")

    one = torch.ones((ctrl.dof,), dtype=torch.float32, device=dev)
    expanded = one.unsqueeze(0).expand(n_envs, -1)  # non-contiguous view
    # should not raise
    ctrl.step(expanded)
    scene.step()


def test_cpu_actions_on_gpu_scene_ok(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointVelocityController(
        entity=robot,
        model=model,
        normalized=True,
        v_max=1.0,
        clip=True,
    )
    ctrl.post_build()

    # on sim device (your base forbids mismatch)
    act = _rand(n_envs, ctrl.dof, device=dev)
    ctrl.step(act)
    scene.step()


def test_joint_torque_normalized_clamps_to_force_range(scene_robot):
    _scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointTorqueController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        tau_max=1.0,
    )
    ctrl.post_build()

    # Ensure no PD contribution when checking applied forces
    zeros = torch.zeros(ctrl.dof, dtype=torch.float32, device=dev)
    robot.set_dofs_kp(zeros, list(range(ctrl.dof)))
    robot.set_dofs_kv(zeros, list(range(ctrl.dof)))

    # Saturate command beyond range
    act = torch.full((n_envs, ctrl.dof), 2.0, dtype=torch.float32, device=dev)
    ctrl.step(act)

    # Read back solver-side forces
    tau = robot.get_dofs_force()
    tau = tau if tau.ndim == 1 else tau[0]
    caps = _per_joint_force_caps(robot, dofs_idx=range(model.dof))
    assert torch.all(tau.abs().cpu() <= caps + 1e-4)


def test_osc_pose_abs_zero_first_action_holds_pose(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
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

    p0 = _eef_pos_2d(robot, model.end_link).detach().clone()
    ctrl.step(_zeros(n_envs, 6, dev))  # zero action captures origin and holds
    scene.step()
    p1 = _eef_pos_2d(robot, model.end_link).detach()

    assert torch.allclose(p0, p1, atol=2e-3)
