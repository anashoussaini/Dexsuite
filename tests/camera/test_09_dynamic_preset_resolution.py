from __future__ import annotations

from types import SimpleNamespace

from dexsuite.core.camera import CameraSystem


class AnonymousGripper:
    """Gripper without a registry name, matching integrated robot behavior."""


def _camera_system() -> CameraSystem:
    return CameraSystem.__new__(CameraSystem)


def test_dynamic_preset_prefers_modular_gripper_override():
    system = _camera_system()
    manipulator = SimpleNamespace(_registry_name="ur5")
    gripper = SimpleNamespace(_registry_name="allegro")

    spec = system._resolve_dynamic_spec("wrist_cam", manipulator, gripper)

    assert spec.pos_offset == (0.0, 0.0, -0.21)
    assert spec.quat_offset == (-0.1830127, -0.6830127, 0.6830127, 0.1830127)


def test_dynamic_preset_uses_manipulator_override_for_integrated_gripper():
    system = _camera_system()
    manipulator = SimpleNamespace(_registry_name="ur5")
    gripper = AnonymousGripper()

    spec = system._resolve_dynamic_spec("wrist_cam", manipulator, gripper)

    assert spec.pos_offset == (0.0, 0.0, 0.15)
    assert spec.quat_offset == (0.3466353, -0.938, 0, 0)


def test_dynamic_preset_accepts_legacy_by_alias():
    preset = {"by": {"robotiq": {"pos_offset": (1.0, 2.0, 3.0)}}}

    overrides = CameraSystem._dynamic_overrides(preset, "legacy")

    assert overrides == {"robotiq": {"pos_offset": (1.0, 2.0, 3.0)}}
