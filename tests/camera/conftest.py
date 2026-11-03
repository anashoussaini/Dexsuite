# tests/camera/conftest.py
# ruff: noqa: N803, S101, S110

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pytest

# Genesis imports `numba` with caching enabled. In sandboxed/CI environments,
# the default user cache directory may be non-writable, causing import-time
# failures. Force the cache into the repo if the user did not specify one.
if "NUMBA_CACHE_DIR" not in os.environ:
    cache_dir = Path(__file__).resolve().parents[2] / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    try:
        import numba.core.config as numba_config

        numba_config.CACHE_DIR = str(cache_dir)
    except Exception:
        pass


_RENDER_BACKED_TESTS = {
    "test_single_default_cameras_rgb_only",
    "test_direct_names_and_all_modalities",
    "test_bimanual_wrist_expansion_rgb_only",
    "test_bimanual_with_all_modalities",
    "test_dataclass_presets_plus_custom_static_res_and_modalities",
    "test_disable_all_cameras_with_none",
    "test_all_modalities_shapes_and_dtypes",
    "test_camera_pack_names_and_modalities_persist_across_steps",
    "test_many_cameras_single_build_shapes_modalities_and_naming",
}


@lru_cache(maxsize=1)
def _offscreen_renderer_available() -> bool:
    """Return whether Genesis can create an offscreen OpenGL context."""
    try:
        from genesis.ext import pyrender

        platform = os.environ.get("PYOPENGL_PLATFORM", "egl")
        renderer = pyrender.OffscreenRenderer(
            pyopengl_platform=platform,
            seg_node_map={},
        )
        renderer.delete()
    except Exception:
        return False
    return True


def pytest_runtest_setup(item) -> None:
    if item.name not in _RENDER_BACKED_TESTS:
        return
    if not _offscreen_renderer_available():
        pytest.skip("Genesis offscreen renderer is unavailable in this environment.")

