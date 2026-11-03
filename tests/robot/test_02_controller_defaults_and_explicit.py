# tests/robot/test_02_controller_defaults_and_explicit.py
import pytest

from dexsuite.options.robot import ArmOptions, ControllerOptions
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


def test_armoptions_injects_yaml_defaults_for_both_roles():
    cfg = load_config("controllers")
    dflt_arm = cfg["defaults"]["manipulator"]
    dflt_grip = cfg["defaults"]["gripper"]
    specs = cfg["controllers"]

    a = ArmOptions()

    assert a.manipulator_controller.name == dflt_arm
    assert a.manipulator_controller.config == specs.get(dflt_arm, {})

    # gripper_controller can be None for integrated robots; for modular defaults,
    # it is present.
    assert a.gripper_controller is None or a.gripper_controller.name == dflt_grip
    if a.gripper_controller is not None:
        assert a.gripper_controller.config == specs.get(dflt_grip, {})


@pytest.mark.parametrize("name", ARM_CTRL_ALL)
def test_controlleroptions_accepts_explicit_arm_names(name):
    c = ControllerOptions(name=name)
    assert c.name == name


@pytest.mark.parametrize("name", GRIP_CTRL_JOINT)
def test_controlleroptions_accepts_explicit_gripper_names(name):
    c = ControllerOptions(name=name)
    assert c.name == name
