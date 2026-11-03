"""Expose robot and scene model definitions and bundled assets.

Model modules are imported by the top-level package so registry decorators can populate manipulator and gripper registries. Bundled object and texture assets are resolved by utility functions rather than imported as Python modules.

Subpackages:
    arena: Arena builders and MJCF arena assets.
    grippers: Gripper and hand model registrations.
    manipulators: Manipulator model registrations.
    objects: Object asset package data.
    textures: Texture asset package data.
"""

from __future__ import annotations
