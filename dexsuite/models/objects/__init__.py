"""Expose bundled object asset metadata.

Object assets are package data resolved by utility functions. Preferred object formats are ordered by `_SUFFIX_PRIORITY` when a caller does not request a specific extension.
"""

from __future__ import annotations

# File type priority when no preference is provided.
_SUFFIX_PRIORITY: tuple[str, ...] = (".xml", ".urdf", ".obj", ".stl", ".glb", ".ply")
