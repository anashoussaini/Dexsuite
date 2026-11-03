# tests/robot/test_03_build_single_parametric.py
import pytest
from _utils import arm_ctrl_field_name

from dexsuite.options.builders import _build_robot_options
from dexsuite.options.robot import ArmOptions, ControllerOptions, RobotOptions
from dexsuite.utils.options_utils import load_config

ARM_CTRL_ALL = (
    "joint_position",
    "joint_velocity",
    "joint_torque",
    "osc_pose",
    "osc_pose_abs",
    "ik_pose",
)
GRIP_CTRL_JOINT = ("joint_position", "joint_velocity", "joint_torque")


def _yaml_default_arm_name():
    return load_config("controllers")["defaults"]["manipulator"]


def _yaml_default_grip_name():
    return load_config("controllers")["defaults"]["gripper"]


@pytest.mark.parametrize("arm_name", (None, *ARM_CTRL_ALL))
@pytest.mark.parametrize("grip_name", (None, *GRIP_CTRL_JOINT))
def test_build_single_with_direct_api(arm_name, grip_name):
    kw = {
        "manipulator": "franka",
        "gripper": "robotiq",
    }
    if arm_name is not None:
        kw["arm_control"] = arm_name
    if grip_name is not None:
        kw["gripper_control"] = grip_name

    rob = _build_robot_options(kw, existing=None)
    assert isinstance(rob, RobotOptions)
    assert rob.type_of_robot == "single"
    assert isinstance(rob.single, ArmOptions)

    ctrl_field = arm_ctrl_field_name()
    arm_ctrl = getattr(rob.single, ctrl_field)
    grip_ctrl = rob.single.gripper_controller

    expected_arm = arm_name if arm_name is not None else _yaml_default_arm_name()
    assert isinstance(arm_ctrl, ControllerOptions)
    assert arm_ctrl.name == expected_arm

    expected_grip = grip_name if grip_name is not None else _yaml_default_grip_name()
    assert isinstance(grip_ctrl, ControllerOptions) or (grip_ctrl is None)
    if isinstance(grip_ctrl, ControllerOptions):
        assert grip_ctrl.name == expected_grip


def test_build_single_preserves_dataclass_defaults_when_not_provided():
    kw = {"manipulator": "franka", "gripper": "robotiq"}
    rob = _build_robot_options(kw, None)
    ctrl_field = arm_ctrl_field_name()
    assert getattr(rob.single, ctrl_field).name == _yaml_default_arm_name()
