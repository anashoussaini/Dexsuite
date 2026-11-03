"""Resolve and materialize MJCF assets for Genesis and MuJoCo loaders.

Paths are resolved against the referring XML file first, then against the
installed `dexsuite` package for package-prefixed references. Materialized files
are cached under `DEXSUITE_MJCF_CACHE_DIR` when set, otherwise in the system
temporary directory.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from xml.etree import ElementTree


def package_path(package: str, *parts: str) -> Path:
    """Return a concrete filesystem path inside an installed package.

    Args:
        package: Importable package name to resolve.
        parts: Path components below the package root.

    Returns:
        Resolved package path as a pathlib object.
    """
    path = files(package)
    for part in parts:
        path = path / part
    return Path(path)


def dexsuite_path(*parts: str) -> Path:
    """Return a concrete filesystem path inside the `dexsuite` package.

    Args:
        parts: Path components below the package root.

    Returns:
        Resolved package path as a pathlib object.
    """
    return package_path("dexsuite", *parts)


def parse_xml_root(path: Path) -> ElementTree.Element:
    """Parse a local XML file and return its root element.

    Defused XML is used when available, otherwise the standard library parser is
    used for package-owned files. Parse failures are normalized to `ValueError`
    so callers can report asset-loading errors consistently.

    Args:
        path: Path to an XML file on disk.

    Returns:
        Parsed XML document root element.

    Raises:
        ValueError: When the XML cannot be parsed.
    """
    try:
        from defusedxml import (
            ElementTree as DefusedElementTree,  # type: ignore[import-not-found]
        )
        from defusedxml.common import (  # type: ignore[import-not-found]
            DefusedXmlException,
        )
    except ImportError:
        try:
            return ElementTree.parse(path).getroot()  # noqa: S314
        except ElementTree.ParseError as exc:
            raise ValueError(f"Invalid MJCF XML: {path}") from exc

    try:
        return DefusedElementTree.parse(path).getroot()
    except (ElementTree.ParseError, DefusedXmlException) as exc:
        raise ValueError(f"Invalid MJCF XML: {path}") from exc


def resolve_mjcf_path(raw: str, *, base_dir: Path) -> Path | None:
    """Resolve an MJCF path string to an existing absolute path.

    Args:
        raw: Raw path value from an MJCF file attribute.
        base_dir: Directory of the XML file containing the path.

    Returns:
        Existing absolute path, or `None` when the path cannot be resolved.
    """
    value = str(raw).strip()
    path = Path(value)
    if path.is_absolute():
        return path if path.exists() else None

    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate

    if value.startswith(("dexsuite/", "dexsuite\\")):
        suffix = value.split("/", 1)[1] if "/" in value else value.split("\\", 1)[1]
        package_candidate = dexsuite_path(*Path(suffix).parts).resolve()
        if package_candidate.exists():
            return package_candidate

    return None


def materialize_mjcf(root_xml: Path) -> Path:
    """Return a merged MJCF file with includes inlined and file paths resolved.

    The output file is content-addressed from the root XML and all included
    files. Existing cache hits are returned without rewriting the cache file.

    Args:
        root_xml: Root MJCF XML file path.

    Returns:
        Path to the materialized MJCF file.
    """
    root_xml = root_xml.resolve()
    deps = _collect_mjcf_includes(root_xml)
    digest = hashlib.sha256()
    for path in deps:
        digest.update(path.read_bytes())

    output = _mjcf_cache_dir() / f"{digest.hexdigest()}.xml"
    if output.exists():
        return output

    mjcf_root = parse_xml_root(root_xml)
    _expand_includes_inplace(mjcf_root, base_dir=root_xml.parent)
    _merge_compilers_inplace(mjcf_root)
    _rewrite_file_attribs_to_abs(mjcf_root, base_dir=root_xml.parent)
    ElementTree.ElementTree(mjcf_root).write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )
    return output


def _mjcf_cache_dir() -> Path:
    override = os.environ.get("DEXSUITE_MJCF_CACHE_DIR")
    path = (
        Path(override).expanduser().resolve()
        if override
        else (Path(tempfile.gettempdir()) / "dexsuite_mjcf").resolve()
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_mjcf_includes(path: Path) -> tuple[Path, ...]:
    seen: set[Path] = set()
    collected: list[Path] = []

    def walk(current: Path) -> None:
        resolved = current.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        collected.append(resolved)
        root = parse_xml_root(resolved)
        for include in root.iter("include"):
            relative = include.attrib.get("file")
            if not relative:
                raise ValueError(f"MJCF include missing 'file' attribute in {resolved}")
            include_path = (resolved.parent / relative).resolve()
            if include_path.exists():
                walk(include_path)

    walk(path)
    return tuple(collected)


def _expand_includes_inplace(root: ElementTree.Element, *, base_dir: Path) -> None:
    index = 0
    while index < len(root):
        child = root[index]
        if child.tag != "include":
            _expand_includes_inplace(child, base_dir=base_dir)
            index += 1
            continue

        relative = child.attrib.get("file")
        if not relative:
            raise ValueError("MJCF include tag is missing a 'file' attribute.")
        include_path = resolve_mjcf_path(relative, base_dir=base_dir)
        if include_path is None:
            raise FileNotFoundError(
                f"MJCF include file not found: '{relative}' (base_dir={base_dir})",
            )

        include_root = parse_xml_root(include_path)
        if include_root.tag != "mujocoinclude":
            raise ValueError(
                f"Expected <mujocoinclude> root in {include_path}, "
                f"got <{include_root.tag}>.",
            )

        root.remove(child)
        insert_at = index
        inserted = 0
        for include_child in list(include_root):
            root.insert(insert_at, include_child)
            _expand_includes_inplace(include_child, base_dir=include_path.parent)
            insert_at += 1
            inserted += 1
        index += inserted


def _merge_compilers_inplace(root: ElementTree.Element) -> None:
    compilers = [child for child in list(root) if child.tag == "compiler"]
    if len(compilers) <= 1:
        return

    merged: dict[str, str] = {}
    for compiler in compilers:
        for key, value in compiler.attrib.items():
            if key not in merged or merged[key] == "":
                merged[key] = value
    for compiler in compilers:
        root.remove(compiler)
    root.insert(0, ElementTree.Element("compiler", attrib=merged))


def _rewrite_file_attribs_to_abs(root: ElementTree.Element, *, base_dir: Path) -> None:
    for element in root.iter():
        raw = element.attrib.get("file")
        if not raw:
            continue
        resolved = resolve_mjcf_path(raw, base_dir=base_dir)
        if resolved is not None:
            element.attrib["file"] = str(resolved)


__all__ = ["dexsuite_path", "materialize_mjcf", "package_path", "parse_xml_root"]
