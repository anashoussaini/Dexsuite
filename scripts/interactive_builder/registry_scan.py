"""Static registries for the interactive builder.

This module avoids importing the dexsuite package at module import time because
importing the main package can trigger simulation initialization in some
environments.

Instead, it scans the DexSuite source tree for decorators of the form
register_env("name"), register_controller("name"), register_manipulator("name"),
and register_gripper("name").
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

_REGISTER_RE = re.compile(
    r"""@register_(?P<kind>env|controller|manipulator|gripper)\(\s*["'](?P<key>[^"']+)["']\s*\)""",
)


def _dexsuite_dir() -> Path:
    """Return the filesystem directory of the dexsuite package sources."""
    spec = find_spec("dexsuite")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed dexsuite package.")
    return Path(next(iter(spec.submodule_search_locations)))


def _iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        if "__pycache__" in p.parts:
            continue
        yield p


def _scan_keys(root: Path, *, kind: str) -> list[str]:
    keys: set[str] = set()
    for path in _iter_py_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Source files should be utf-8; skip if not.
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = _REGISTER_RE.search(s)
            if not m:
                continue
            if m.group("kind") == kind:
                keys.add(m.group("key"))
    return sorted(keys)


def available_tasks() -> list[str]:
    """Return all environment task keys (from @register_env)."""
    return _scan_keys(_dexsuite_dir() / "environments", kind="env")


def available_controllers() -> list[str]:
    """Return all controller keys (from @register_controller)."""
    return _scan_keys(_dexsuite_dir() / "controllers", kind="controller")


def available_manipulators() -> list[str]:
    """Return all manipulator model keys (from @register_manipulator)."""
    return _scan_keys(_dexsuite_dir() / "models" / "manipulators", kind="manipulator")


def available_grippers() -> list[str]:
    """Return all gripper model keys (from @register_gripper)."""
    return _scan_keys(_dexsuite_dir() / "models" / "grippers", kind="gripper")


@dataclass(frozen=True)
class ManipulatorInfo:
    """Static information about a manipulator model."""

    key: str
    kind: str  # "integrated" | "modular" | "unknown"
    file: Path
    cls_name: str | None


def _base_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _decorator_key(deco: ast.expr, expected_name: str) -> str | None:
    if not isinstance(deco, ast.Call):
        return None
    if isinstance(deco.func, ast.Name):
        name = deco.func.id
    elif isinstance(deco.func, ast.Attribute):
        name = deco.func.attr
    else:
        return None
    if name != expected_name:
        return None
    if not deco.args:
        return None
    v = deco.args[0]
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value
    return None


def _infer_manipulator_kind(base_names: Iterable[str]) -> str:
    bases = {b for b in base_names if b}
    if "IntegratedManipulatorModel" in bases:
        return "integrated"
    if "ModularManipulatorModel" in bases:
        return "modular"
    return "unknown"


def manipulator_infos() -> list[ManipulatorInfo]:
    """Return ManipulatorInfo for registered manipulators.

    Notes:
        This is a static AST-based scan. It does not import any simulation
        dependencies and will classify a manipulator as:
        - "integrated" if it inherits IntegratedManipulatorModel
        - "modular" if it inherits ModularManipulatorModel
        - "unknown" otherwise
    """
    root = _dexsuite_dir() / "models" / "manipulators"
    infos: dict[str, ManipulatorInfo] = {}
    for path in _iter_py_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            keys = [
                k
                for k in (
                    _decorator_key(deco, "register_manipulator")
                    for deco in node.decorator_list
                )
                if k is not None
            ]
            if not keys:
                continue
            base_names = [_base_name(b) for b in node.bases]
            kind = _infer_manipulator_kind(base_names)
            for k in keys:
                infos[k] = ManipulatorInfo(
                    key=k,
                    kind=kind,
                    file=path,
                    cls_name=node.name,
                )
    return [infos[k] for k in sorted(infos)]


def supported_grippers_for_manipulator(manipulator_key: str) -> list[str] | None:
    """Best-effort list of grippers supported by a modular manipulator.

    This reads <name>_adapters.yaml files under dexsuite/models/manipulators/<name>/
    when present and returns the top-level mapping keys (gripper names).

    Returns:
        - list[str]: gripper keys if an adapters YAML is found and parsed.
        - None: if no adapters file exists (unknown compatibility).
    """
    infos = {i.key: i for i in manipulator_infos()}
    info = infos.get(manipulator_key)
    if info is None:
        return None
    search_dir = info.file.parent
    candidates = sorted(search_dir.glob("*_adapters.yaml"))
    if not candidates:
        return None

    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read adapters YAML files (pip install pyyaml).",
        ) from e

    for path in candidates:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return sorted(str(k) for k in data)
    return None


def available_static_cameras() -> list[str]:
    """Return static camera preset names from config/cameras.yaml."""
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read cameras.yaml (pip install pyyaml).",
        ) from e
    path = _dexsuite_dir() / "config" / "cameras.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    static = cfg.get("static") if isinstance(cfg, dict) else None
    if not isinstance(static, dict):
        return []
    return sorted(static.keys())
