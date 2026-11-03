from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from dexsuite.controllers import _lazy_ensure
from dexsuite.core.registry import CONTROLLER_REG, GRIP_REG, MANIP_REG
from dexsuite.utils.options_utils import load_workspaces

importlib.import_module("dexsuite.models.grippers")
importlib.import_module("dexsuite.models.manipulators")

CONFIG_ROOT = Path("dexsuite/config")
LAYOUT_NAMES = {"single", "side_by_side", "face_to_face", "right_angle"}
MOUNT_GEOMETRY_FIELDS = {
    "collision",
    "fixed",
    "height",
    "radius",
    "size",
    "visualization",
}


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_all_dexsuite_yaml_files_parse() -> None:
    for path in sorted(Path("dexsuite").rglob("*.yaml")):
        _read_yaml(path)


def test_config_yaml_has_no_tabs_or_trailing_whitespace() -> None:
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert "\t" not in line, f"{path}:{line_number} contains a tab"
            assert line == line.rstrip(), f"{path}:{line_number} has trailing whitespace"


def test_workspaces_match_registered_manipulators() -> None:
    workspaces = load_workspaces()
    assert set(workspaces) == set(MANIP_REG)


def test_camera_overrides_match_registered_robot_names() -> None:
    cameras = _read_yaml(CONFIG_ROOT / "cameras.yaml")
    overrides = (((cameras or {}).get("dynamic") or {}).get("wrist_cam") or {}).get(
        "overrides",
        {},
    )
    known_names = set(MANIP_REG) | set(GRIP_REG)
    assert set(overrides) <= known_names


def test_controller_config_matches_registered_controllers() -> None:
    configured_names = {
        "joint_position",
        "joint_velocity",
        "joint_torque",
        "osc_pose",
        "osc_pose_abs",
        "osc_pose_abs_quat",
        "ik_pose",
    }
    for name in configured_names:
        _lazy_ensure(name)

    controllers = _read_yaml(CONFIG_ROOT / "controllers.yaml")
    sections = set((controllers or {}).get("controllers") or {})
    assert sections == set(CONTROLLER_REG)


def test_layout_files_exist_for_every_registered_manipulator() -> None:
    expected = set(MANIP_REG) | {"defaults"}
    actual = {path.stem for path in (CONFIG_ROOT / "layouts").glob("*.yaml")}
    assert actual == expected

    for layout_name in expected:
        layout = _read_yaml(CONFIG_ROOT / "layouts" / f"{layout_name}.yaml")
        assert layout.get("name") == layout_name
        assert set(layout) >= LAYOUT_NAMES


def test_adapter_geometry_contains_only_geometry_fields() -> None:
    for path in sorted(Path("dexsuite/models/manipulators").glob("*/*_adapters.yaml")):
        adapters = _read_yaml(path) or {}
        for name, adapter in adapters.items():
            geometry = adapter.get("geometry", {}) if isinstance(adapter, dict) else {}
            if not isinstance(geometry, dict):
                continue
            extra = set(geometry) - MOUNT_GEOMETRY_FIELDS
            assert not extra, f"{path}:{name}.geometry has non-geometry keys {extra}"
