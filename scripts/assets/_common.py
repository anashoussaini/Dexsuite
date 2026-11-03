"""Shared bootstrap for the asset generators (``python scripts/assets/make_<asset>.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.assets.artiverse_import import (  # noqa: E402
    OBJECTS_ROOT,
    Box,
    Cylinder,
    JointSpec,
    LinkSpec,
    attribution_text,
    find_model_urdf,
    import_model,
)

__all__ = ["OBJECTS_ROOT", "Box", "Cylinder", "JointSpec", "LinkSpec", "attribution_text", "find_model_urdf", "import_model"]
