from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.robots.integrated_robot import _HandProxy


class _ExplicitIntegratedArm(IntegratedManipulatorModel):
    model_path = __file__  # must exist; the model is never loaded
    dof = 6
    control_dofs_index: ClassVar[list[int]] = [0, 2, 3, 5]
    root_link = "base"
    end_link = "tool"
    home_q = torch.zeros(4, dtype=torch.float32)
    builtin_gripper_dof = 2
    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [1, 4]
    builtin_gripper_root_link = "hand"
    builtin_gripper_home_q = torch.tensor([0.1, 0.2], dtype=torch.float32)
    kp = torch.arange(6, dtype=torch.float32)


class _TailIntegratedArm(IntegratedManipulatorModel):
    model_path = __file__  # must exist; the model is never loaded
    dof = 5
    control_dofs_index: ClassVar[list[int]] = [0, 1, 2]
    root_link = "base"
    end_link = "tool"
    home_q = torch.zeros(3, dtype=torch.float32)
    builtin_gripper_dof = 2
    builtin_gripper_control_dofs_index = None
    builtin_gripper_root_link = "hand"


class _BadIntegratedArm(IntegratedManipulatorModel):
    model_path = __file__  # must exist; the model is never loaded
    dof = 5
    control_dofs_index: ClassVar[list[int]] = [0, 1, 2]
    root_link = "base"
    end_link = "tool"
    home_q = torch.zeros(3, dtype=torch.float32)
    builtin_gripper_dof = 2
    builtin_gripper_control_dofs_index: ClassVar[list[int]] = [4]
    builtin_gripper_root_link = "hand"


def test_integrated_gripper_uses_explicit_dof_indices():
    arm = _ExplicitIntegratedArm()

    assert arm.gripper_dofs_index == (1, 4)

    hand = _HandProxy(arm, torch.device("cpu"))
    assert hand.control_dofs_index == [1, 4]
    assert hand.dof == 2
    assert torch.equal(hand.kp, torch.tensor([1.0, 4.0], device=hand.kp.device))


def test_integrated_gripper_falls_back_to_tail_indices():
    arm = _TailIntegratedArm()

    assert arm.gripper_dofs_index == (3, 4)

    hand = _HandProxy(arm, torch.device("cpu"))
    assert hand.control_dofs_index == [3, 4]


def test_integrated_gripper_index_length_must_match_dof():
    arm = _BadIntegratedArm()

    with pytest.raises(ValueError, match="builtin_gripper_control_dofs_index length"):
        _ = arm.gripper_dofs_index


def test_integrated_gripper_no_finger_slice_api():
    arm = _ExplicitIntegratedArm()

    assert not hasattr(arm, "finger_slice")
