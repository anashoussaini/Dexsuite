"""Measured progress for spraying a loose circle around a pane's centre."""

from __future__ import annotations

import math

import numpy as np


class PaneSprayProgress:
    """Count actual spray reaching a closed pane, without a path-shape goal."""

    def __init__(self, *, minimum_steps=20):
        self.minimum_steps = int(minimum_steps)
        self.steps = 0

    def reset(self):
        self.steps = 0

    def update(self, *, closed, active, hit_uv):
        if not closed:
            self.reset()
            return
        if not active:
            return
        hits = np.asarray(hit_uv, dtype=float).reshape(-1, 2)
        if not np.isfinite(hits).all():
            raise ValueError('Nonfinite measured pane spray intersection')
        if len(hits) and np.any(np.all((hits >= 0.) & (hits <= 1.), axis=1)):
            self.steps += 1

    @property
    def success(self):
        return self.steps >= self.minimum_steps


class CircularSprayProgress:
    """Count directions actually sprayed after the sash has been lowered.

    The nozzle's centre ray must aim in the central annulus and real spray rays
    must hit nearby glass. A stationary spray, line, or a circle elsewhere does
    not visit the full set of angular sectors. Radius and direction are free
    inside the generous annulus; this is not a precision tracing task.
    """

    def __init__(self, pane_size, *, radius_min=.07, radius_max=.14,
                 sectors=24, required_sectors=22, minimum_steps=24):
        self.pane_size = np.asarray(pane_size, dtype=float)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.required_sectors = int(required_sectors)
        self.minimum_steps = int(minimum_steps)
        self.visited = np.zeros(int(sectors), dtype=bool)
        self.steps = 0

    def reset(self):
        self.visited[:] = False
        self.steps = 0

    def update(self, *, closed, active, axis_uv, hit_uv):
        if not closed:
            self.reset()
            return
        if not active or axis_uv is None:
            return
        axis = (np.asarray(axis_uv, dtype=float).reshape(2) - .5) * self.pane_size
        hits = (np.asarray(hit_uv, dtype=float).reshape(-1, 2) - .5) * self.pane_size
        if not np.isfinite(axis).all() or not np.isfinite(hits).all():
            raise ValueError('Nonfinite measured circular spray intersection')
        radius = float(np.linalg.norm(axis))
        if not self.radius_min <= radius <= self.radius_max:
            return
        # Actual droplets, rather than the aim ray alone, must reach the glass
        # close to the intended circular stroke. The existing cone/splat model
        # determines the visible paint, without any synthetic target markings.
        if not len(hits) or not np.any(np.linalg.norm(hits-axis, axis=1) <= .05):
            return
        angle = math.atan2(float(axis[1]), float(axis[0])) % (2*math.pi)
        sector = min(int(angle / (2*math.pi) * len(self.visited)), len(self.visited)-1)
        self.visited[sector] = True
        self.steps += 1

    @property
    def coverage(self):
        return float(self.visited.mean())

    @property
    def success(self):
        return bool(self.visited.sum() >= self.required_sectors and self.steps >= self.minimum_steps)


def spray_axis_uv(origin, direction, pane_position, pane_rotation, pane_size, *, near, far, face_offset=0.0005):
    """Intersect the actual nozzle axis with the same face as PaintCanvas."""
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3).copy()
    rotation = np.asarray(pane_rotation, dtype=float).reshape(3, 3)
    position = np.asarray(pane_position, dtype=float).reshape(3)
    if not all(np.isfinite(x).all() for x in (origin, direction, rotation, position)):
        raise ValueError('Nonfinite nozzle or pane pose')
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return None
    direction /= length
    normal = rotation[:, 2]
    face = position + face_offset * normal
    denom = float(direction @ normal)
    if denom >= -1e-6:
        return None
    distance = float((face-origin) @ normal) / denom
    if not near <= distance <= far:
        return None
    local = rotation.T @ (origin + distance*direction-face)
    uv = local[:2] / np.asarray(pane_size, dtype=float) + .5
    return uv if np.all((uv >= 0.) & (uv <= 1.)) else None
