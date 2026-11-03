from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from scripts.interactive_builder.runner import (
    _DexsuiteDictDeviceSource,
    _loop,
    _pack_single_action,
)


@dataclass
class _Seg:
    start: int
    stop: int


def _fake_env(
    *,
    arm_dim: int,
    grip_dim: int,
    home_q,
    q_lo,
    q_hi,
    normalized: bool = True,
    open_q=None,
    closed_q=None,
):
    layout = SimpleNamespace(
        segments=(
            _Seg(0, arm_dim),
            _Seg(arm_dim, arm_dim + grip_dim),
        ),
    )
    hand_ctrl = SimpleNamespace(
        normalized=normalized,
        _lo_1d=torch.tensor(q_lo, dtype=torch.float32),
        _hi_1d=torch.tensor(q_hi, dtype=torch.float32),
    )
    hand_ent = SimpleNamespace(
        get_dofs_position=lambda: torch.tensor(home_q, dtype=torch.float32),
    )
    gripper = SimpleNamespace(
        home_q=torch.tensor(home_q, dtype=torch.float32),
        open_q=(None if open_q is None else torch.tensor(open_q, dtype=torch.float32)),
        closed_q=(
            None if closed_q is None else torch.tensor(closed_q, dtype=torch.float32)
        ),
    )
    robot = SimpleNamespace(
        _layout=layout,
        device=torch.device("cpu"),
        hand_ctrl=hand_ctrl,
        hand_ent=hand_ent,
        gripper=gripper,
    )
    return SimpleNamespace(robot=robot)


class _FakeDevice:
    def __init__(self, actions):
        self._actions = list(actions)

    def get_action(self):
        if not self._actions:
            raise RuntimeError("No more fake actions.")
        return self._actions.pop(0)


class _OneStepSource:
    def __init__(self, action):
        self._action = action
        self.calls = 0

    def next_action(self):
        self.calls += 1
        if self.calls > 1:
            raise KeyboardInterrupt
        return self._action

    def close(self):
        return None


class _DoneEnv:
    def __init__(self, *, info):
        self.n_envs = 1
        self.reset_calls = 0
        self._info = info

    def step(self, _action):
        return (
            {},
            torch.tensor([0.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            self._info,
        )

    def reset(self):
        self.reset_calls += 1
        return {}, {}


def _expected(values, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def test_pack_single_action_expands_scalar_to_mirrored_gripper_targets():
    env = _fake_env(
        arm_dim=6,
        grip_dim=2,
        home_q=[0.04, -0.04],
        q_lo=[0.0, -0.04],
        q_hi=[0.04, 0.0],
    )
    pose = torch.zeros(6, dtype=torch.float32)

    open_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([-1.0]))
    close_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([1.0]))

    assert torch.allclose(open_action[-2:], _expected([1.0, -1.0], open_action))
    assert torch.allclose(close_action[-2:], _expected([-1.0, 1.0], close_action))


def test_pack_single_action_keeps_same_sign_for_standard_parallel_gripper():
    env = _fake_env(
        arm_dim=6,
        grip_dim=2,
        home_q=[0.04, 0.04],
        q_lo=[0.0, 0.0],
        q_hi=[0.04, 0.04],
    )
    pose = torch.zeros(6, dtype=torch.float32)

    open_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([-1.0]))
    close_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([1.0]))

    assert torch.allclose(open_action[-2:], _expected([1.0, 1.0], open_action))
    assert torch.allclose(close_action[-2:], _expected([-1.0, -1.0], close_action))


def test_pack_single_action_uses_explicit_robotiq_open_closed_targets():
    env = _fake_env(
        arm_dim=6,
        grip_dim=1,
        home_q=[0.0],
        q_lo=[0.0],
        q_hi=[0.85],
        open_q=[0.0],
        closed_q=[0.85],
    )
    pose = torch.zeros(6, dtype=torch.float32)

    open_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([-1.0]))
    close_action = _pack_single_action(env, pose=pose, gripper=torch.tensor([1.0]))

    assert torch.allclose(open_action[-1:], _expected([-1.0], open_action))
    assert torch.allclose(close_action[-1:], _expected([1.0], close_action))


def test_device_source_holds_last_gripper_target_on_zero_release():
    env = _fake_env(
        arm_dim=6,
        grip_dim=1,
        home_q=[0.0],
        q_lo=[0.0],
        q_hi=[0.85],
        open_q=[0.0],
        closed_q=[0.85],
    )
    device = _FakeDevice(
        [
            {"pose": [0.0] * 6, "gripper": [1.0]},
            {"pose": [0.0] * 6, "gripper": [0.0]},
        ],
    )
    src = _DexsuiteDictDeviceSource(env, device_obj=device)

    close_action = src.next_action()
    hold_action = src.next_action()

    assert torch.allclose(close_action[-1:], _expected([1.0], close_action))
    assert torch.allclose(hold_action[-1:], _expected([1.0], hold_action))


def test_loop_does_not_double_reset_dexsuite_auto_reset_env():
    env = _DoneEnv(info={"needs_reset": torch.tensor([True])})
    src = _OneStepSource(torch.zeros(1, dtype=torch.float32))

    _loop(env, SimpleNamespace(control_hz=1000), src)

    assert env.reset_calls == 0


def test_loop_resets_gym_style_done_env_without_auto_reset_signal():
    env = _DoneEnv(info={})
    src = _OneStepSource(torch.zeros(1, dtype=torch.float32))

    _loop(env, SimpleNamespace(control_hz=1000), src)

    assert env.reset_calls == 1
