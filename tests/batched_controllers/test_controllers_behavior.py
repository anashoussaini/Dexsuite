from __future__ import annotations

from pathlib import Path

import genesis as gs
import pytest
import torch

from dexsuite.controllers.joint_position import JointPositionController
from dexsuite.controllers.joint_torque import JointTorqueController
from dexsuite.controllers.joint_velocity import JointVelocityController
from dexsuite.controllers.osc_pose import OSCPoseController
from dexsuite.controllers.osc_pose_abs import OSCPoseAbsoluteController
from dexsuite.controllers.osc_pose_abs_quat import OSCPoseAbsoluteQuatController

try:
    from dexsuite.controllers.ik_pose import IKPoseController

    HAS_IK = True
except ImportError:
    HAS_IK = False


def _ones(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D
    return torch.ones((n_envs, cols), dtype=torch.float32, device=device)


def _zeros(n_envs: int, cols: int, device: torch.device) -> torch.Tensor:
    # Always 2D
    return torch.zeros((n_envs, cols), dtype=torch.float32, device=device)


def _eef_pos(robot, link_name="hand"):
    link = robot.get_link(link_name)
    pos = link.get_pos()
    return pos if pos.ndim == 1 else pos[0]


def _qpos_2d(robot) -> torch.Tensor:
    qpos = robot.get_dofs_position()
    return qpos.unsqueeze(0) if qpos.ndim == 1 else qpos


def _qvel_2d(robot) -> torch.Tensor:
    qvel = robot.get_dofs_velocity()
    return qvel.unsqueeze(0) if qvel.ndim == 1 else qvel


def _eef_pos_2d(robot, link_name="hand") -> torch.Tensor:
    pos = robot.get_link(link_name).get_pos()
    return pos.unsqueeze(0) if pos.ndim == 1 else pos


def _run_controller(scene, ctrl, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        ctrl.step(action)
        scene.step()


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANDA_XML = _REPO_ROOT / "dexsuite" / "models" / "manipulators" / "franka" / "assets" / "panda.xml"


def test_set_home_position_sets_qpos(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointPositionController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
    )
    ctrl.post_build()

    # Move away from home a bit
    ctrl.step(_ones(n_envs, ctrl.dof, dev) * 0.1)
    scene.step()

    # Now set home
    ctrl.set_home_position()

    q = robot.get_dofs_position()
    q = q if q.ndim == 1 else q[0]
    home = torch.as_tensor(model.home_q, dtype=torch.float32, device=q.device)
    # Slightly relaxed tolerance to account for sim timing
    assert torch.allclose(q, home, atol=1e-3)


def test_joint_position_command_moves_first_joint(scene_robot):
    scene, robot, _ee, model, _n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointPositionController(
        entity=robot,
        model=model,
        normalized=False,
        clip=True,
    )
    ctrl.post_build()

    q0 = _qpos_2d(robot).detach().clone()
    target = q0[:, : ctrl.dof].clone()
    target[:, 0] += 0.05

    _run_controller(scene, ctrl, target.to(dev), steps=25)

    q1 = _qpos_2d(robot).detach()
    assert torch.all(q1[:, 0] > q0[:, 0] + 1e-3)


def test_joint_velocity_command_moves_first_joint(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointVelocityController(
        entity=robot,
        model=model,
        normalized=False,
        clip=True,
        v_max=0.6,
    )
    ctrl.post_build()

    q0 = _qpos_2d(robot).detach().clone()
    action = _zeros(n_envs, ctrl.dof, dev)
    action[:, 0] = 0.4

    _run_controller(scene, ctrl, action, steps=15)

    q1 = _qpos_2d(robot).detach()
    assert torch.all(q1[:, 0] > q0[:, 0] + 1e-4)


def test_joint_torque_command_accelerates_first_joint(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = JointTorqueController(
        entity=robot,
        model=model,
        normalized=False,
        clip=True,
        tau_max=1.0,
    )
    ctrl.post_build()

    zeros = torch.zeros(ctrl.dof, dtype=torch.float32, device=dev)
    robot.set_dofs_kp(zeros, ctrl.dofs_idx)
    robot.set_dofs_kv(zeros, ctrl.dofs_idx)

    action = _zeros(n_envs, ctrl.dof, dev)
    action[:, 0] = 5.0

    _run_controller(scene, ctrl, action, steps=10)

    qvel = _qvel_2d(robot).detach()
    assert torch.all(qvel[:, 0] > 1e-5)


@pytest.mark.parametrize("steps,threshold", [(10, 1e-4), (20, 2e-4)])
def test_osc_pose_small_delta_moves_eef_forward(scene_robot, steps, threshold):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    # ang_scale must be > 0 per your controller; use tiny positive nonzero
    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        lin_scale=0.01,
        ang_scale=1e-6,
    )
    ctrl.post_build()

    p0 = _eef_pos(robot).detach().clone()
    # positive X translation command
    act = _zeros(n_envs, 6, dev)
    act[:, 0] = 1.0  # x

    for _ in range(steps):
        ctrl.step(act)
        scene.step()

    p1 = _eef_pos(robot).detach().clone()
    dx = (p1 - p0)[0].item()
    assert dx > threshold


@pytest.mark.parametrize("steps,threshold", [(10, 5e-5), (20, 1e-4)])
def test_osc_pose_panda_mjcf_small_delta_moves_eef_forward(scene_robot, steps, threshold):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    model.model_path = str(_PANDA_XML)

    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        lin_scale=0.01,
        ang_scale=1e-6,
        mink_dt=0.05,
        max_joint_step=0.05,
    )
    ctrl.post_build()

    p0 = _eef_pos(robot).detach().clone()
    act = _zeros(n_envs, 6, dev)
    act[:, 0] = 1.0

    for _ in range(steps):
        ctrl.step(act)
        scene.step()

    p1 = _eef_pos(robot).detach().clone()
    dx = (p1 - p0)[0].item()
    assert dx > threshold


def test_osc_pose_partial_reset_keeps_other_env_state(scene_robot):
    _scene, robot, _ee, model, n_envs = scene_robot
    if n_envs < 2:
        pytest.skip("partial reset is only meaningful for batched scenes")

    dev = torch.device(gs.device)
    model.model_path = str(_PANDA_XML)
    ctrl = OSCPoseController(
        entity=robot,
        model=model,
        normalized=True,
        clip=True,
        lin_scale=0.01,
        ang_scale=1e-6,
        mink_dt=0.05,
        max_joint_step=0.05,
    )
    ctrl.post_build()

    action = _zeros(n_envs, 6, dev)
    action[0, 0] = 1.0
    action[1, 1] = -1.0
    ctrl.step(action)

    old_pos_ref = ctrl._pos_ref.detach().clone()
    old_quat_ref = ctrl._quat_ref.detach().clone()
    old_twist_ref = ctrl._twist_ref.detach().clone()
    old_state_0 = ctrl._backend._states[0]
    old_state_1 = ctrl._backend._states[1]

    ctrl.reset(env_idx=torch.tensor([0], dtype=torch.long))

    assert ctrl._ref_valid is not None
    assert not bool(ctrl._ref_valid[0].item())
    assert bool(torch.all(ctrl._ref_valid[1:]).item())
    assert torch.allclose(ctrl._pos_ref[1:], old_pos_ref[1:])
    assert torch.allclose(ctrl._quat_ref[1:], old_quat_ref[1:])
    assert torch.allclose(ctrl._twist_ref[1:], old_twist_ref[1:])
    assert ctrl._backend._states[0] is not old_state_0
    assert ctrl._backend._states[1] is old_state_1


def test_osc_pose_abs_offset_moves_eef_forward(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = OSCPoseAbsoluteController(
        entity=robot,
        model=model,
        normalized=False,
        clip=False,
        lin_range=(0.05, 0.05, 0.05),
        ang_range=(0.1, 0.1, 0.1),
        mink_dt=0.05,
        max_joint_step=0.05,
    )
    ctrl.post_build()

    p0 = _eef_pos_2d(robot).detach().clone()
    action = _zeros(n_envs, 6, dev)
    action[:, 0] = 0.03

    _run_controller(scene, ctrl, action, steps=20)

    p1 = _eef_pos_2d(robot).detach()
    assert torch.all(p1[:, 0] > p0[:, 0] + 5e-5)


@pytest.mark.parametrize(
    ("controller_cls", "action_cols", "extra_kwargs"),
    [
        (
            OSCPoseAbsoluteController,
            6,
            {"ang_range": (0.1, 0.1, 0.1)},
        ),
        (
            OSCPoseAbsoluteQuatController,
            7,
            {},
        ),
    ],
)
def test_absolute_pose_partial_reset_keeps_other_env_origin(
    scene_robot,
    controller_cls,
    action_cols,
    extra_kwargs,
):
    _scene, robot, _ee, model, n_envs = scene_robot
    if n_envs < 2:
        pytest.skip("partial reset is only meaningful for batched scenes")

    dev = torch.device(gs.device)
    model.model_path = str(_PANDA_XML)
    ctrl = controller_cls(
        entity=robot,
        model=model,
        normalized=False,
        clip=False,
        lin_range=(0.05, 0.05, 0.05),
        mink_dt=0.05,
        max_joint_step=0.05,
        **extra_kwargs,
    )
    ctrl.post_build()

    action = _zeros(n_envs, action_cols, dev)
    if action_cols == 7:
        action[:, 3] = 1.0
    ctrl.step(action)

    old_origin_pos = ctrl._origin_pos.detach().clone()
    old_origin_quat = ctrl._origin_quat.detach().clone()
    old_state_0 = ctrl._backend._states[0]
    old_state_1 = ctrl._backend._states[1]

    ctrl.reset(env_idx=torch.tensor([0], dtype=torch.long))

    assert ctrl._origin_valid is not None
    assert not bool(ctrl._origin_valid[0].item())
    assert bool(torch.all(ctrl._origin_valid[1:]).item())
    assert torch.allclose(ctrl._origin_pos[1:], old_origin_pos[1:])
    assert torch.allclose(ctrl._origin_quat[1:], old_origin_quat[1:])
    assert ctrl._backend._states[0] is not old_state_0
    assert ctrl._backend._states[1] is old_state_1


def test_osc_pose_abs_quat_offset_moves_eef_forward(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = OSCPoseAbsoluteQuatController(
        entity=robot,
        model=model,
        normalized=False,
        clip=False,
        lin_range=(0.05, 0.05, 0.05),
        mink_dt=0.05,
        max_joint_step=0.05,
    )
    ctrl.post_build()

    p0 = _eef_pos_2d(robot).detach().clone()
    action = _zeros(n_envs, 7, dev)
    action[:, 0] = 0.03
    action[:, 3] = 1.0

    _run_controller(scene, ctrl, action, steps=20)

    p1 = _eef_pos_2d(robot).detach()
    assert torch.all(p1[:, 0] > p0[:, 0] + 5e-5)


@pytest.mark.skipif(not HAS_IK, reason="IKPoseController unavailable")
def test_ik_pose_absolute_target_moves_eef_forward(scene_robot):
    scene, robot, _ee, model, n_envs = scene_robot
    dev = torch.device(gs.device)
    ctrl = IKPoseController(
        entity=robot,
        model=model,
        normalized=False,
        clip=False,
        ik_max_iters=80,
        ik_damping=0.05,
        ik_max_step=0.05,
        ik_init="current",
        orientation_cost=0.0,
    )
    ctrl.post_build()

    p0 = _eef_pos_2d(robot).detach().clone()
    action = _zeros(n_envs, 6, dev)
    action[:, :3] = p0.to(dev)
    action[:, 0] += 0.03

    _run_controller(scene, ctrl, action, steps=20)

    p1 = _eef_pos_2d(robot).detach()
    assert torch.all(p1[:, 0] > p0[:, 0] + 5e-5)
