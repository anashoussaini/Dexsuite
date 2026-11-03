# tests/robot/test_05_layout_resolution_yaml.py

import pytest

from dexsuite.options import ArmOptions, RobotOptions
from dexsuite.options.layout import LayoutOptions, PoseOptions


def test_single_layout_preset_is_rejected():
    with pytest.raises(ValueError):
        RobotOptions(
            type_of_robot="single",
            single=ArmOptions(manipulator="franka", gripper="robotiq"),
            layout=LayoutOptions(preset="single"),  # forbidden for single-arm by design
        )


def test_single_layout_explicit_pose_allowed():
    rob = RobotOptions(
        type_of_robot="single",
        single=ArmOptions(manipulator="franka", gripper="robotiq"),
        layout=LayoutOptions(single=PoseOptions(pos=(0.2, 0.3, 0.0), yaw_rad=0.5)),
    )
    assert rob.layout.single is not None
    assert tuple(rob.layout.single.pos) == (0.2, 0.3, 0.0)
    assert abs(rob.layout.single.yaw_rad - 0.5) < 1e-9
