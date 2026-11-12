"""Keep one renderer's teardown from breaking every other renderer in the process.

A process that builds a second env with cameras while an earlier one is still around breaks the live env's cameras
when the earlier env is freed (by ``close()`` or by the garbage collector), in three ways inside Genesis 0.4.3:

1. ``Rasterizer.destroy`` deletes its camera render targets before it makes its own context current, so their GL
   names are deleted in whichever context is current, usually the live env's (GL names are per context): the live
   env's framebuffers go and its next render fails (``glBindFramebuffer``, GL_INVALID_OPERATION).
2. The EGL platform's ``delete_context`` calls ``eglTerminate`` on the display, which EGL shares between all the
   contexts of a device in a process: every other context dies with it (``EGL_NOT_INITIALIZED``).
3. Renderers render without making their context current again, so once a teardown has switched contexts the live
   env renders into the wrong one.

`patch_genesis_egl_teardown` fixes the three: a rasterizer is torn down in its own context and the previously
current context is restored afterwards; a display is terminated only with its last context; an offscreen renderer
makes its context current before it renders. It is applied once, when DexSuite initialises Genesis.
"""

from __future__ import annotations

import ctypes
from collections import Counter
from typing import Any

_LIVE: Counter = Counter()  # live contexts per EGL display


def _handle(value: Any) -> int | None:
    """A hashable identity for an EGL handle (a ctypes pointer)."""
    if value is None:
        return None
    try:
        return ctypes.cast(value, ctypes.c_void_p).value
    except (TypeError, ctypes.ArgumentError):
        raw = getattr(value, "value", None)
        return raw if isinstance(raw, int) else None


def _patch_platform(egl_module) -> None:
    platform = egl_module.EGLPlatform
    original_init = platform.init_context

    def init_context(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        key = _handle(getattr(self, "_egl_display", None))
        if key is not None:
            _LIVE[key] += 1

    def delete_context(self):
        from OpenGL.EGL import eglDestroyContext, eglTerminate

        display = getattr(self, "_egl_display", None)
        if display is None:
            return
        if getattr(self, "_egl_context", None) is not None:
            eglDestroyContext(display, self._egl_context)
            self._egl_context = None
        key = _handle(display)
        if key is None:  # unknown handle: Genesis's behaviour
            eglTerminate(display)
        else:
            _LIVE[key] -= 1
            if _LIVE[key] <= 0:
                del _LIVE[key]
                eglTerminate(display)
        self._egl_display = None

    platform.init_context = init_context
    platform.delete_context = delete_context


def _patch_rasterizer(rasterizer_module) -> None:
    rasterizer = rasterizer_module.Rasterizer
    original_destroy = rasterizer.destroy

    def destroy(self):
        from OpenGL import EGL

        renderer = getattr(self, "_renderer", None)
        own = (
            getattr(renderer, "_platform", None)
            if getattr(self, "_offscreen", False)
            else None
        )
        if own is None or getattr(own, "_egl_context", None) is None:
            return original_destroy(self)
        previous = (
            EGL.eglGetCurrentDisplay(),
            EGL.eglGetCurrentSurface(EGL.EGL_DRAW),
            EGL.eglGetCurrentSurface(EGL.EGL_READ),
            EGL.eglGetCurrentContext(),
        )
        own_context = _handle(own._egl_context)
        own.make_current()  # the camera targets are deleted in the context that owns them
        try:
            return original_destroy(self)
        finally:
            if (
                _handle(previous[3]) not in (None, 0, own_context)
                and _handle(previous[0]) in _LIVE
            ):
                EGL.eglMakeCurrent(*previous)

    rasterizer.destroy = destroy


def _patch_offscreen(offscreen_module) -> None:
    offscreen = offscreen_module.OffscreenRenderer
    original_render = offscreen.render

    def render(self, *args, **kwargs):
        platform = getattr(self, "_platform", None)
        if platform is not None and getattr(platform, "_egl_context", None) is not None:
            platform.make_current()
        return original_render(self, *args, **kwargs)

    offscreen.render = render


def patch_genesis_egl_teardown() -> bool:
    """Patch Genesis's offscreen EGL rendering so freeing one env leaves the other envs' cameras working.

    Returns:
        True when the patch is in place, False when this install has no EGL rendering to patch.
    """
    try:
        from genesis.ext.pyrender import offscreen as offscreen_module
        from genesis.ext.pyrender.platforms import egl as egl_module
        from genesis.vis import rasterizer as rasterizer_module
    except Exception:  # noqa: BLE001 - no EGL/OpenGL in this install, nothing to patch
        return False
    if getattr(egl_module.EGLPlatform, "_dexsuite_teardown_patch", False):
        return True
    _patch_platform(egl_module)
    _patch_rasterizer(rasterizer_module)
    _patch_offscreen(offscreen_module)
    egl_module.EGLPlatform._dexsuite_teardown_patch = True
    return True
