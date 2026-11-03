from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from dexsuite.core.registry import ENV_REG
from dexsuite.environments.base_env import BaseEnv
from dexsuite.utils.viewer_utils import set_viewer_shadows

NO_VIEWER_SHADOWS = {"put_bread_toaster", "spray_graffiti", "open_safe", "close_window_spray", "pliers_in_jewelry_box"}


class _FakePyrenderViewer:
    """Mimics genesis.ext.pyrender.Viewer: one render_flags dict shared by both passes."""

    def __init__(self) -> None:
        self.render_flags = {"shadows": True, "offscreen": False}
        self.seen: list[tuple[str, bool]] = []

    def _render(self, *args, **kwargs):
        self.seen.append(("offscreen" if self.render_flags["offscreen"] else "screen", self.render_flags["shadows"]))
        return args

    def on_draw(self):
        return self._render()

    def render_offscreen(self, camera_node, target):
        self.render_flags["offscreen"] = True
        try:
            return self._render(camera_node, target)
        finally:
            self.render_flags["offscreen"] = False


def _scene(viewer: _FakePyrenderViewer | None) -> SimpleNamespace:
    if viewer is None:
        return SimpleNamespace(viewer=None)
    return SimpleNamespace(viewer=SimpleNamespace(_pyrender_viewer=viewer))


def test_no_viewer_is_a_noop() -> None:
    assert set_viewer_shadows(_scene(None), False) is False
    assert set_viewer_shadows(SimpleNamespace(viewer=SimpleNamespace(_pyrender_viewer=None)), False) is False


def test_screen_pass_loses_shadows_camera_pass_keeps_them() -> None:
    viewer = _FakePyrenderViewer()
    assert set_viewer_shadows(_scene(viewer), False) is True
    viewer.on_draw()
    assert viewer.render_offscreen("cam", "target") == ("cam", "target")
    viewer.on_draw()
    assert viewer.seen == [("screen", False), ("offscreen", True), ("screen", False)]
    assert viewer.render_flags["shadows"] is True


def test_flag_is_restored_even_when_the_render_raises() -> None:
    viewer = _FakePyrenderViewer()

    def boom(*args, **kwargs):
        raise RuntimeError("gl")

    viewer._render = boom
    set_viewer_shadows(_scene(viewer), False)
    with pytest.raises(RuntimeError):
        viewer.on_draw()
    assert viewer.render_flags["shadows"] is True


def test_idempotent_and_reversible() -> None:
    viewer = _FakePyrenderViewer()
    set_viewer_shadows(_scene(viewer), False)
    wrapped = viewer._render
    set_viewer_shadows(_scene(viewer), False)
    assert viewer._render is wrapped
    set_viewer_shadows(_scene(viewer), True)
    assert viewer._render is wrapped
    viewer.on_draw()
    viewer.render_offscreen("cam", "target")
    assert viewer.seen == [("screen", True), ("offscreen", True)]
    set_viewer_shadows(_scene(viewer), False)
    viewer.on_draw()
    assert viewer.seen[-1] == ("screen", False)


def test_composes_with_a_later_render_wrapper() -> None:
    viewer = _FakePyrenderViewer()
    set_viewer_shadows(_scene(viewer), False)
    inner = viewer._render
    calls: list[int] = []

    def paint_then_render(*args, **kwargs):
        calls.append(1)
        return inner(*args, **kwargs)

    viewer._render = paint_then_render
    viewer.on_draw()
    viewer.render_offscreen("cam", "target")
    assert len(calls) == 2
    assert viewer.seen == [("screen", False), ("offscreen", True)]


def test_only_the_articulated_envs_opt_out() -> None:
    importlib.import_module("dexsuite.environments")
    assert BaseEnv.VIEWER_SHADOWS is True
    for name in NO_VIEWER_SHADOWS:
        assert ENV_REG[name].VIEWER_SHADOWS is False, name
    assert ENV_REG["pick_place_pan"].VIEWER_SHADOWS is True
