# tests/robot/test_06_errors.py
import pytest

from dexsuite import api
from dexsuite.core.robots.factory import make_arm_from_options
from dexsuite.options import ArmOptions, PoseOptions
from dexsuite.options.builders import _build_robot_options


def test_missing_manipulator_raises():
    with pytest.raises(ValueError):
        _ = _build_robot_options({}, existing=None)


def test_bimanual_gripper_wrong_length_raises():
    with pytest.raises(ValueError):
        _ = _build_robot_options(
            {
                "manipulator": ("franka", "franka"),
                "gripper": ("robotiq",),  # wrong length
            },
            existing=None,
        )


def test_make_unknown_task_fails_before_genesis_init(monkeypatch):
    def fail_init(**_kw):
        raise AssertionError("Genesis init should not run for unknown tasks")

    monkeypatch.setattr(api, "_lazy_init_genesis", fail_init)

    with pytest.raises(ValueError, match="Unknown task"):
        api.make("__missing_task__", manipulator="franka", gripper="robotiq")


def test_make_non_string_task_raises_type_error(monkeypatch):
    def fail_init(**_kw):
        raise AssertionError("Genesis init should not run for invalid task types")

    monkeypatch.setattr(api, "_lazy_init_genesis", fail_init)

    with pytest.raises(TypeError, match="task must be a string"):
        api.make(123, manipulator="franka", gripper="robotiq")


def test_make_passes_performance_mode_to_genesis_init(monkeypatch):
    calls = []

    class DummyEnv:
        def __init__(self, robot, cameras, sim, render_mode):
            self.robot = robot
            self.cameras = cameras
            self.sim = sim
            self.render_mode = render_mode

    def record_init(*, performance_mode, precision):
        calls.append((performance_mode, precision))

    monkeypatch.setitem(api.ENV_REG, "dummy_task", DummyEnv)
    monkeypatch.setattr(api, "_lazy_init_genesis", record_init)

    api.make(
        "dummy_task",
        manipulator="franka",
        gripper="robotiq",
        cameras=None,
        performance_mode=True,
    )

    if calls != [(True, "32")]:
        raise AssertionError(f"Expected Genesis performance_mode=True, got {calls}")


def test_modular_arm_requires_explicit_gripper():
    with pytest.raises(ValueError, match="requires a gripper"):
        make_arm_from_options(
            arm_options=ArmOptions(manipulator="franka", gripper=None),
            scene=object(),
            pose_options=PoseOptions(),
        )


def test_modular_arm_unknown_gripper_lists_known_grippers():
    with pytest.raises(ValueError, match="Available grippers"):
        make_arm_from_options(
            arm_options=ArmOptions(manipulator="franka", gripper="not_a_gripper"),
            scene=object(),
            pose_options=PoseOptions(),
        )
