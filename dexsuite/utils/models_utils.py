"""Resolve DexSuite asset paths for models, objects, and textures."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

_SUFFIX_PRIORITY: tuple[str, ...] = (".xml", ".urdf", ".obj", ".stl", ".glb", ".ply")

# The dexsuite package on PyPI ships every robot and the assets of every task but Close Window and Spray (its window model is
# over PyPI's file-size limit); a clone also has that task's models and the object models no task loads.
MISSING_ASSET_HINT = (
    "This asset is not in the dexsuite package, which ships every robot and the assets of every task but Close Window "
    "and Spray (its window model is over PyPI's file-size limit). That task and the rest of the object library come "
    "with the repository: https://github.com/anashoussaini/Dexsuite"
)


def _models_path(*parts: str) -> Path:
    path = files("dexsuite.models")
    for part in parts:
        path = path / part
    return Path(path)


def package_resource_path(package: str | None, *parts: str) -> str:
    """Return a filesystem path for a resource in an installed package.

    Args:
        package: Package name used to resolve resources.
        parts: Resource path components to append.
    """
    if package is None:
        raise RuntimeError("Package resource paths require package execution context.")
    path = files(package)
    for part in parts:
        path = path / part
    return str(Path(path))


def package_asset_path(package: str | None, *parts: str) -> str:
    """Return a filesystem path under a package's assets directory.

    Args:
        package: Package name used to resolve resources.
        parts: Resource path components to append.
    """
    return package_resource_path(package, "assets", *parts)


def root(name: str) -> Path:
    """Return the root directory path for an asset type.

    Checks for a DEXSUITE_<NAME>_DIR environment variable override first.
    If not found, it defaults to the models/<name> directory relative
    to this file's location.

    Args:
        name: The asset category name (e.g., "objects", "textures").

    Returns:
        The resolved, absolute path to the asset root directory.
    """
    override = os.environ.get(f"DEXSUITE_{name.upper()}_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _models_path(name).resolve()


def get_texture_paths(name: str) -> dict[str, str]:
    """Finds all texture component files associated with a given texture name.

    This function searches for a directory under the textures root and maps
    each file to a texture component based on its name. For example, a file
    named wood_planks_normal.png in the wood_planks directory would be
    mapped to the "normal" key. A file named wood_planks.png is assumed to
    be the "diffuse" map.

    Args:
        name: The name of the texture, corresponding to a directory under the textures root (e.g., "wood_planks").

    Returns:
        A dictionary mapping texture component names (e.g., "diffuse", "normal")
        to filesystem paths as strings.

    Raises:
        FileNotFoundError: If the specified texture directory does not exist.
    """
    asset_root = root("textures")
    texture: dict[str, str] = {}
    direct = (asset_root / name).resolve()
    if direct.exists() and direct.is_dir():
        for file in direct.iterdir():
            if file.is_file():
                key = file.stem.split("_")[-1]
                target = name.split("/")[-1]
                if key == target:
                    texture["diffuse"] = str(file)
                else:
                    texture[key] = str(file)
        return texture
    raise FileNotFoundError(f"Texture '{name}' not found under '{asset_root}'. {MISSING_ASSET_HINT}")


def _ordered_exts(prefer: str | None) -> tuple[str, ...]:
    """Returns an ordered tuple of extensions based on an optional preference.

    Examples:
        _ordered_exts("urdf") -> (".urdf", ".xml", ".obj", ...)
        _ordered_exts("mesh") -> (".obj", ".stl", ".glb", ...)

    Args:
        prefer: The preferred file type. Can be a specific extension (e.g., "obj"), a category ("mesh"), or an alias ("mjcf").

    Returns:
        A tuple of file extensions, ordered by preference.
    """
    if not prefer:
        return _SUFFIX_PRIORITY

    prefer_key = prefer.lower().lstrip(".")
    if prefer_key == "mjcf":
        prefer_key = "xml"

    preferred = (
        [".obj", ".stl", ".glb", ".ply"] if prefer_key == "mesh" else [f".{prefer_key}"]
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for ext in preferred + list(_SUFFIX_PRIORITY):
        if ext in _SUFFIX_PRIORITY and ext not in seen:
            ordered.append(ext)
            seen.add(ext)
    return tuple(ordered)


def get_object_path(name: str, prefer: str | None = None) -> Path:
    """Resolves and returns the primary file path for an object asset.

    This function finds an object file using the following resolution rules:
    1.  If name is a direct path to a file, it is returned.
    2.  If name is a directory, it searches for a file within that directory,
        preferring a file that matches the directory's name (e.g., mug/mug.obj).
    3.  If name is a stem, it searches for a file with a matching extension
        at the root level (e.g., mug.obj).

    Args:
        name: The object identifier. Can be a nested path (e.g., "kitchen/mug"), a directory, or a direct filename.
        prefer: An optional preferred file type: "mjcf", "urdf", "mesh", or a concrete extension like "xml" or "obj".

    Returns:
        The resolved, absolute path to the object file.

    Raises:
        FileNotFoundError: If no matching object file can be found.
        ValueError: If the search within a directory is ambiguous due to
            multiple candidate files.
    """
    asset_root = root("objects")

    direct = (asset_root / name).resolve()
    if direct.exists() and direct.is_file():
        return direct

    base_dir = (asset_root / name).resolve()
    if base_dir.exists() and base_dir.is_dir():
        exts = _ordered_exts(prefer)

        for ext in exts:
            cand = base_dir / f"{base_dir.name}{ext}"
            if cand.exists():
                return cand

        all_hits: list[Path] = []
        for ext in exts:
            hits = sorted(base_dir.glob(f"*{ext}"))
            if len(hits) == 1:
                return hits[0]
            all_hits.extend(hits)

        if all_hits:
            listed = "\n  ".join(str(p) for p in all_hits)
            raise ValueError(
                f"Ambiguous object '{name}', multiple candidates:\n  {listed}",
            )

        raise FileNotFoundError(f"No known object file under '{base_dir}'")

    for ext in _ordered_exts(prefer):
        cand = (asset_root / f"{name}{ext}").resolve()
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Object '{name}' not found under '{asset_root}'. {MISSING_ASSET_HINT}")
