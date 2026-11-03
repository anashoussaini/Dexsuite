"""Interactive-viewer-only rendering tweaks; recorded camera images are left untouched."""

from __future__ import annotations

_STATE_ATTR = "_dexsuite_viewer_shadows"


def set_viewer_shadows(scene, enabled: bool) -> bool:
    """Show or hide cast shadows in the interactive viewer window only.

    Genesis draws the on-screen frame and every camera image (``viewer.render_offscreen``)
    through the same ``pyrender.Viewer._render`` and the same ``render_flags["shadows"]``
    entry; ``render_flags["offscreen"]`` is what tells the two passes apart. The on-screen
    pass is wrapped so the flag is cleared for it alone, under the viewer's render lock,
    and restored afterwards, so camera renders keep ``VisOptions.shadow``.

    Args:
        scene: Built ``gs.Scene``. Without a live viewer (render_mode "rgb_array"/None)
            this is a no-op.
        enabled: False hides viewer shadows, True shows them again.

    Returns:
        True when a viewer was configured, False when there is none. Idempotent.
    """
    viewer = getattr(getattr(scene, "viewer", None), "_pyrender_viewer", None)
    if viewer is None:
        return False
    state = getattr(viewer, _STATE_ATTR, None)
    if state is None:
        state = {"enabled": True}
        original_render = viewer._render

        def _render(*args, **kwargs):
            flags = viewer.render_flags
            if state["enabled"] or flags["offscreen"]:
                return original_render(*args, **kwargs)
            saved = flags["shadows"]
            flags["shadows"] = False
            try:
                return original_render(*args, **kwargs)
            finally:
                flags["shadows"] = saved

        setattr(viewer, _STATE_ATTR, state)
        viewer._render = _render
    state["enabled"] = bool(enabled)
    return True
