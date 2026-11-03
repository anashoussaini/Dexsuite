# tests/robot/test_04_build_bimanual_parametric.py
import pytest
from _utils import arm_ctrl_field_name

from dexsuite.options.builders import _build_robot_options
from dexsuite.options.robot import ControllerOptions
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
@pytest.mark.parametrize("gripper_param", ["single", "pair"])
def test_build_bimanual_with_direct_api(arm_name, grip_name, gripper_param):
    kw = {
        "manipulator": ("franka", "franka"),
        "gripper": ("robotiq", "robotiq") if gripper_param == "pair" else "robotiq",
        "layout_preset": "face_to_face",
    }
    if arm_name is not None:
        kw["arm_control"] = arm_name
    if grip_name is not None:
        kw["gripper_control"] = grip_name

    rob = _build_robot_options(kw, None)
    assert rob.type_of_robot == "bimanual"
    ctrl_field = arm_ctrl_field_name()

    for arm in (rob.left, rob.right):
        ac = getattr(arm, ctrl_field)
        gc = arm.gripper_controller

        expected_arm = arm_name if arm_name is not None else _yaml_default_arm_name()
        assert isinstance(ac, ControllerOptions)
        assert ac.name == expected_arm

        expected_grip = (
            grip_name if grip_name is not None else _yaml_default_grip_name()
        )
        assert isinstance(gc, ControllerOptions) or (gc is None)
        if isinstance(gc, ControllerOptions):
            assert gc.name == expected_grip
