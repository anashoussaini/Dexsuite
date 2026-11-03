"""A paintable textured quad (paper, glass pane, ...) whose texture is updated in place at runtime.

The canvas is a UV-mapped 1 x 1 x 0.001 m slab (``task/paint_quad``) scaled to ``size``; its local +Z
face is the paintable front face. Paint lives in a numpy mask (H, W) and is composited into an RGBA image that is
pushed to the rasterizer's OpenGL texture with ``glTexSubImage2D`` (depth / segmentation untouched).
With the offscreen rasterizer (``render_mode="rgb_array"``) the upload happens immediately; with the viewer
(``render_mode="human"``, the teleop collectors) it is queued and flushed on the viewer thread right before its
next frame, so recorded camera frames still see the paint. Only if neither path exists does the canvas fall back
to ``scene.draw_debug_spheres`` (real geometry, visible in depth; see :attr:`uses_debug`).

An optional stencil (:meth:`PaintCanvas.set_stencil`) draws a coloured outline of a target region into the base
image (under the paint, redrawn on every :meth:`PaintCanvas.reset`) and lets :meth:`PaintCanvas.coverage` score the
paint against the region's interior. Without a stencil the canvas behaves exactly as before.

Single-env feature: batched scenes share one texture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import genesis as gs
import numpy as np
import torch

from dexsuite.utils.models_utils import get_object_path
from dexsuite.utils.robot_utils import link_entities

_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0])
FACE_OFFSET = 0.0005  # the paintable +z face of the 1 mm slab


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _gl_upload(tex, image: np.ndarray) -> None:
    """Write ``image`` into the pyrender texture in place; the GL context must be current on this thread."""
    from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glGenerateMipmap, glTexSubImage2D

    tex.source = image
    if tex._in_context():
        data = np.ascontiguousarray(np.flip(image, axis=0)).reshape(-1)
        tex._bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.shape[1], image.shape[0], GL_RGBA, GL_UNSIGNED_BYTE, data)
        glGenerateMipmap(GL_TEXTURE_2D)
        tex._unbind()


def _to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(t) else np.asarray(t, dtype=np.float64)


def _shift(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``mask`` moved by (dr, dc) texels, zero-filled."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    r0, r1 = max(dr, 0), min(H + dr, H)
    c0, c1 = max(dc, 0), min(W + dc, W)
    out[r0:r1, c0:c1] = mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _disc_offsets(radius: int) -> list[tuple[int, int]]:
    r = int(radius)
    return [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1) if 0 < dr * dr + dc * dc <= r * r]


def stencil_outline(interior: np.ndarray, width_px: int) -> np.ndarray:
    """Boolean ring about ``width_px`` texels wide straddling the boundary of ``interior`` (H, W bool)."""
    interior = np.asarray(interior, dtype=bool)
    r_out = max(int(math.ceil(0.5 * int(width_px))), 1)
    r_in = max(int(width_px) - r_out, 0)
    dilated, eroded = interior.copy(), interior.copy()
    for dr, dc in _disc_offsets(r_out):
        dilated |= _shift(interior, dr, dc)
    for dr, dc in _disc_offsets(r_in):
        eroded &= _shift(interior, dr, dc)
    return dilated & ~eroded


def region_coverage(mask: np.ndarray, interior: np.ndarray | None, threshold: float, region: str | None) -> float:
    """Paint statistics of ``mask`` (H, W float in [0, 1]) against a stencil ``interior`` (H, W bool or None).

    ``region=None``: fraction of all texels above ``threshold``. ``"inside"``: fraction of the interior texels above
    ``threshold``. ``"outside"``: share of the paint mass (sum of the mask) lying outside the interior, 0 when
    nothing is painted.
    """
    if region is None:
        return float((mask > threshold).mean())
    if interior is None:
        raise ValueError("coverage(region=...) needs a stencil; call set_stencil() first")
    if region == "inside":
        return float((mask[interior] > threshold).mean()) if interior.any() else 0.0
    if region == "outside":
        total = float(mask.sum())
        return float(mask[~interior].sum() / total) if total > 0.0 else 0.0
    raise ValueError(f"unknown coverage region {region!r} (None, 'inside' or 'outside')")


class PaintCanvas:
    """Paintable quad. Build in ``_setup_scene``; call :meth:`reset` in ``_on_episode_start``."""

    def __init__(
        self,
        scene,
        *,
        size: Sequence[float],
        res: Sequence[int] = (512, 512),
        base_rgb: Sequence[int] = (245, 245, 245),
        paint_rgb: Sequence[int] = (25, 25, 25),
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        attach_to=None,
        parent_link_name: str = "",
        max_debug_points: int = 4000,
    ) -> None:
        self.scene = scene
        self.size = (float(size[0]), float(size[1]))
        self.height, self.width = int(res[0]), int(res[1])
        self._base = np.asarray(base_rgb, dtype=np.float32)
        self._paint = np.asarray(paint_rgb, dtype=np.float32)
        self._base_image: np.ndarray | None = None  # (H, W, 3) base with the stencil outline drawn in
        self._stencil: np.ndarray | None = None
        self._outline: np.ndarray | None = None
        self.mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.image = np.empty((self.height, self.width, 4), dtype=np.uint8)
        self.image[..., 3] = 255
        self._compose()
        self._attached = attach_to is not None
        self._use_debug = False
        self._debug_node = None
        self._debug_points: list[np.ndarray] = []
        self._max_debug_points = int(max_debug_points)
        self._warned = False
        self._rng = np.random.default_rng(0)

        self.entity = scene.add_entity(
            gs.morphs.Mesh(
                file=str(get_object_path("task/paint_quad")),
                scale=(self.size[0], self.size[1], 1.0),
                pos=tuple(float(v) for v in pos),
                quat=tuple(float(v) for v in quat),
                fixed=True,
                collision=False,
                visualization=True,
                batch_fixed_verts=self._attached,
            ),
            material=gs.materials.Rigid(rho=50.0),
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_array=self.image.copy()),
                double_sided=True,
            ),
        )
        if self._attached:
            link_entities(
                parent_entity=attach_to,
                child_entity=self.entity,
                parent_link_name=parent_link_name,
                child_link_name=self.entity.links[0].name,
            )

    # ------------------------------------------------------------------ pose
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World (pos (3,), rotation matrix (3, 3)) of the canvas centre (env 0 when batched)."""
        pos = _to_np(self.entity.get_pos())
        quat = _to_np(self.entity.get_quat())
        if pos.ndim == 2:
            pos, quat = pos[0], quat[0]
        return pos, _quat_to_mat(quat)

    def normal(self) -> np.ndarray:
        _, rot = self.pose()
        return rot @ _LOCAL_NORMAL

    # ------------------------------------------------------------------ hits
    def spray_hits(
        self,
        origin,
        direction,
        *,
        half_angle_rad: float,
        n_rays: int = 32,
        near: float = 0.02,
        far: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intersect a random cone of rays with the front face.

        Returns ``(uv (M, 2) in [0, 1]^2, points_world (M, 3))`` for the rays that hit within
        ``[near, far]`` while travelling against the face normal.
        """
        rng = rng or self._rng
        o = _to_np(origin).reshape(3)
        d = _to_np(direction).reshape(3)
        d = d / max(np.linalg.norm(d), 1e-9)
        helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, helper)
        e1 /= max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(d, e1)
        theta = float(half_angle_rad) * np.sqrt(rng.random(n_rays))
        phi = 2.0 * math.pi * rng.random(n_rays)
        dirs = (
            np.cos(theta)[:, None] * d
            + np.sin(theta)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        )
        pos, rot = self.pose()
        n = rot @ _LOCAL_NORMAL
        pos = pos + FACE_OFFSET * n
        denom = dirs @ n
        valid = denom < -1e-6
        s_hit = ((pos - o) @ n) / np.where(valid, denom, -1.0)
        valid &= (s_hit >= near) & (s_hit <= far)
        pts = o + s_hit[:, None] * dirs
        local = (pts - pos) @ rot  # rot^T applied row-wise
        uv = np.stack([local[:, 0] / self.size[0] + 0.5, local[:, 1] / self.size[1] + 0.5], axis=1)
        valid &= (uv >= 0.0).all(axis=1) & (uv <= 1.0).all(axis=1)
        return uv[valid], pts[valid]

    # ------------------------------------------------------------------ paint
    def splat(self, uv, radius_m: float, strength: float = 0.6) -> int:
        """Add Gaussian blobs (sigma = radius / 2) at the given UVs; returns the number of blobs."""
        uv = _to_np(uv).reshape(-1, 2)
        if uv.shape[0] == 0:
            return 0
        H, W = self.height, self.width
        sigma = max(0.5 * float(radius_m) * self.px_per_m, 0.75)
        win = int(math.ceil(2.5 * sigma))
        ys = np.arange(-win, win + 1, dtype=np.float32)
        gy, gx = np.meshgrid(ys, ys, indexing="ij")
        kernel = (np.exp(-(gx**2 + gy**2) / (2.0 * sigma**2)) * float(strength)).astype(np.float32)
        cols = np.rint(uv[:, 0] * W).astype(int)
        rows = np.rint((1.0 - uv[:, 1]) * H).astype(int)  # OBJ v = 0 is the bottom image row
        for r, c in zip(rows.tolist(), cols.tolist()):
            r0, r1 = max(r - win, 0), min(r + win + 1, H)
            c0, c1 = max(c - win, 0), min(c + win + 1, W)
            if r0 >= r1 or c0 >= c1:
                continue
            k = kernel[r0 - (r - win) : r1 - (r - win), c0 - (c - win) : c1 - (c - win)]
            np.minimum(self.mask[r0:r1, c0:c1] + k, 1.0, out=self.mask[r0:r1, c0:c1])
        return int(uv.shape[0])

    def coverage(self, threshold: float = 0.5, region: str | None = None) -> float:
        """Fraction of texels whose paint mask exceeds ``threshold``; see :func:`region_coverage` for ``region``."""
        return region_coverage(self.mask, self._stencil, threshold, region)

    @property
    def px_per_m(self) -> float:
        return 0.5 * (self.width / self.size[0] + self.height / self.size[1])

    # ------------------------------------------------------------------ stencil
    def set_stencil(
        self,
        interior: np.ndarray,
        *,
        outline_rgb: Sequence[int] = (220, 30, 30),
        outline_m: float = 0.004,
        outline_px: int | None = None,
    ) -> None:
        """Score paint against ``interior`` (H, W bool) and draw its outline into the base image.

        The outline is a ring ``outline_px`` texels wide (default: ``outline_m`` converted with the canvas resolution)
        centred on the interior boundary; it sits under the paint and is redrawn by :meth:`reset`.
        """
        interior = np.asarray(interior, dtype=bool)
        if interior.shape != (self.height, self.width):
            raise ValueError(f"stencil shape {interior.shape} != canvas ({self.height}, {self.width})")
        if outline_px is None:
            outline_px = max(int(round(float(outline_m) * self.px_per_m)), 1)
        outline = stencil_outline(interior, int(outline_px))
        base = np.broadcast_to(self._base, (self.height, self.width, 3)).copy()
        base[outline] = np.asarray(outline_rgb, dtype=np.float32)
        self._stencil, self._outline, self._base_image = interior, outline, base
        if self.entity.is_built:
            self.upload()

    def clear_stencil(self) -> None:
        self._stencil = self._outline = self._base_image = None
        if self.entity.is_built:
            self.upload()

    @property
    def stencil(self) -> np.ndarray | None:
        """Interior mask (H, W bool) of the current stencil, or None."""
        return self._stencil

    @property
    def outline(self) -> np.ndarray | None:
        """Outline ring (H, W bool) drawn into the base image, or None."""
        return self._outline

    def reset(self) -> None:
        self.mask.fill(0.0)
        self._debug_points.clear()
        if self._debug_node is not None:
            try:
                self.scene.clear_debug_object(self._debug_node)
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._debug_node = None
        if self.entity.is_built:
            self.upload()

    @property
    def uses_debug(self) -> bool:
        """True once the canvas has fallen back to debug spheres (viewer mode or no offscreen rasterizer)."""
        return self._use_debug

    # ------------------------------------------------------------------ upload
    def _compose(self) -> None:
        m = self.mask[..., None]
        base = self._base if self._base_image is None else self._base_image
        rgb = base * (1.0 - m) + self._paint * m
        self.image[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)

    def upload(self, new_points_world=None) -> None:
        """Push the current mask to the GPU texture (or to debug spheres on the fallback path)."""
        if new_points_world is not None and self._use_debug:
            self._debug_points.append(np.asarray(new_points_world, dtype=np.float32).reshape(-1, 3))
        if self._use_debug:
            self._redraw_debug()
            return
        self._compose()
        try:
            self._upload_texture()
        except Exception as exc:  # noqa: BLE001 - private renderer API
            if not self._warned:
                gs.logger.warning(f"PaintCanvas: texture update failed ({exc!r}).")
                self._warned = True
            if not isinstance(exc, (RuntimeError, KeyError, AttributeError)):
                return  # transient GL error: keep the mask, retry on the next upload
            self._use_debug = True  # no offscreen rasterizer / no node: fall back for good
            if new_points_world is not None:
                self._debug_points.append(np.asarray(new_points_world, dtype=np.float32).reshape(-1, 3))
            self._redraw_debug()

    def _upload_texture(self) -> None:
        vis = self.scene.visualizer
        node = vis.context.rigid_nodes[self.entity.vgeoms[0].uid]
        tex = node.mesh.primitives[0].material.baseColorTexture
        if tex is None:
            raise RuntimeError("canvas material has no base color texture")
        image = self.image.copy()
        renderer = vis.rasterizer._renderer
        if renderer is not None:  # offscreen rasterizer (render_mode="rgb_array"): the GL context is ours to take
            renderer.make_current()
            try:
                _gl_upload(tex, image)
            finally:
                renderer.make_uncurrent()
            return
        viewer = getattr(vis.viewer, "_pyrender_viewer", None)
        if viewer is None or not viewer.is_active:
            raise RuntimeError("no offscreen rasterizer and no live viewer")
        if not viewer.run_in_thread:
            viewer.switch_to()
            _gl_upload(tex, image)
            return
        # Threaded viewer: GL calls must run on the viewer thread. Queue the upload and flush it right before the
        # viewer renders its next frame (on-screen or camera), where the context is current and render_lock is held.
        pending = getattr(viewer, "_gs_paint_uploads", None)
        if pending is None:
            original_render = viewer._render
            pending = {}

            def _render_with_paint(*args, **kwargs):
                # popitem is atomic, so an env-thread insert during the drain is never lost; a failing upload must
                # not escape (it would kill the viewer loop and leave render_offscreen waiting forever)
                while pending:
                    _, (texture, img) = pending.popitem()
                    try:
                        _gl_upload(texture, img)
                    except Exception as exc:  # noqa: BLE001
                        viewer._gs_paint_error = exc
                return original_render(*args, **kwargs)

            viewer._gs_paint_uploads = pending
            viewer._render = _render_with_paint
        error = getattr(viewer, "_gs_paint_error", None)
        if error is not None:
            raise RuntimeError(f"viewer-thread texture upload failed: {error!r}")
        pending[id(tex)] = (tex, image)

    def _redraw_debug(self) -> None:
        if self._debug_node is not None:
            try:
                self.scene.clear_debug_object(self._debug_node)
            except Exception:  # noqa: BLE001
                pass
            self._debug_node = None
        if not self._debug_points:
            return
        pts = np.concatenate(self._debug_points, axis=0)
        if pts.shape[0] > self._max_debug_points:
            pts = pts[np.linspace(0, pts.shape[0] - 1, self._max_debug_points).astype(int)]
            self._debug_points = [pts]
        color = (*(self._paint / 255.0).tolist(), 1.0)
        radius = 0.35 * self.size[0] / self.width * 8.0
        self._debug_node = self.scene.draw_debug_spheres(pts, radius=radius, color=color)
