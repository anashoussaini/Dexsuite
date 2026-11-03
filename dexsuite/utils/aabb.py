"""Provide torch-based axis-aligned bounding boxes.

Bounds are stored as float32 tensors and support batched `(B, 3)` or unbatched `(3,)` shapes. Coordinate values are expressed in the caller's frame, with transform helpers producing world-frame boxes when given base poses.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from dexsuite.utils.globals import get_device
from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch

TensorLike = Sequence[float] | torch.Tensor


@dataclass(frozen=True)
class AABB:
    """Represent an axis-aligned bounding box backed by torch tensors.

    Bounds are normalized so `lo <= hi` elementwise. Methods preserve batching whenever inputs are batched.

    Attributes:
        lo: Lower bounds as a float32 tensor shaped `(3,)` or `(B, 3)`.
        hi: Upper bounds as a float32 tensor shaped `(3,)` or `(B, 3)`.
    """

    min: torch.Tensor
    max: torch.Tensor
    device: torch.device

    @staticmethod
    def from_lists(lo: Iterable[float], hi: Iterable[float]) -> AABB:
        """Create an AABB from Python lists/iterables.

        Args:
            lo: Value (x, y, z) lower bounds.
            hi: Value (x, y, z) upper bounds.

        Returns:
            AABB: A new AABB instance with min/max on the default device.
        """
        dev = get_device()
        lo_t = torch.as_tensor(lo, dtype=torch.float32, device=dev).reshape(3)
        hi_t = torch.as_tensor(hi, dtype=torch.float32, device=dev).reshape(3)
        return AABB(
            min=torch.minimum(lo_t, hi_t),
            max=torch.maximum(lo_t, hi_t),
            device=dev,
        )

    @staticmethod
    def from_tensors(lo: torch.Tensor, hi: torch.Tensor) -> AABB:
        """Create an AABB from torch tensors.

        Args:
            lo: Value (3,) lower bound tensor.
            hi: Value (3,) upper bound tensor.

        Returns:
            AABB: A new AABB instance with min/max on lo.device.
        """
        dev = lo.device
        lo = lo.to(dtype=torch.float32, device=dev).reshape(3)
        hi = hi.to(dtype=torch.float32, device=dev).reshape(3)
        return AABB(min=torch.minimum(lo, hi), max=torch.maximum(lo, hi), device=dev)

    def corners(self) -> torch.Tensor:
        """Return the 8 corner points.

        Returns:
            torch.Tensor: (8, 3) corners on this AABB's device.
        """
        lo, hi = self.min, self.max
        return torch.stack(
            [
                torch.stack([lo[0], lo[1], lo[2]]),
                torch.stack([lo[0], lo[1], hi[2]]),
                torch.stack([lo[0], hi[1], lo[2]]),
                torch.stack([lo[0], hi[1], hi[2]]),
                torch.stack([hi[0], lo[1], lo[2]]),
                torch.stack([hi[0], lo[1], hi[2]]),
                torch.stack([hi[0], hi[1], lo[2]]),
                torch.stack([hi[0], hi[1], hi[2]]),
            ],
            dim=0,
        ).to(dtype=torch.float32, device=self.device)

    def center(self) -> torch.Tensor:
        """Compute the center of the AABB.

        Returns:
            torch.Tensor: (3,) center point.
        """
        return 0.5 * (self.min + self.max)

    def extent(self) -> torch.Tensor:
        """Compute side lengths.

        Returns:
            torch.Tensor: (3,) lengths (max - min).
        """
        return self.max - self.min

    def transform_by_base(
        self,
        base_pos: TensorLike,
        base_quat_wxyz: TensorLike,
    ) -> AABB:
        """Create an enclosing AABB after applying a rigid transform.

        Args:
            base_pos: Value (3,) translation.
            base_quat_wxyz: Value (4,) rotation as WXYZ quaternion.

        Returns:
            AABB: Enclosing AABB after transform.
        """
        rotmat = quat_to_rotmat_wxyz_torch(
            torch.as_tensor(base_quat_wxyz, dtype=torch.float32, device=self.device),
        )
        t = torch.as_tensor(base_pos, dtype=torch.float32, device=self.device).reshape(
            3,
        )
        cs = (self.corners() @ rotmat.T) + t
        return AABB(
            min=cs.min(dim=0).values,
            max=cs.max(dim=0).values,
            device=self.device,
        )

    def with_margin(self, margin: float | Sequence[float]) -> AABB:
        """Shrink (positive margin) or expand (negative margin) uniformly or per-axis.

        Args:
            margin: Scalar or (mx, my, mz) in meters.

        Returns:
            AABB: Margin-adjusted AABB (never inverted).
        """
        m = torch.as_tensor(margin, dtype=torch.float32, device=self.device)
        if m.ndim == 0:
            m = m.repeat(3)
        lo = self.min + m
        hi = self.max - m
        cen = 0.5 * (self.min + self.max)
        lo = torch.minimum(lo, cen)
        hi = torch.maximum(hi, cen)
        return AABB(min=lo, max=hi, device=self.device)

    def contains(self, xyz: TensorLike) -> torch.Tensor:
        """Check if a point or batch of points are inside/on the boundary.

        Args:
            xyz: Value (3,) point or (B, 3) batch of points.

        Returns:
            torch.Tensor: (B,) bool mask. For a single point input (3,), this is a
            1-element tensor.
        """
        p = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        p = p.view(1, 3) if p.ndim == 1 else p.view(-1, 3)
        return ((p >= self.min) & (p <= self.max)).all(dim=1)

    def clamp(self, xyz: TensorLike) -> torch.Tensor:
        """Clamp a point or batch of points to lie inside the AABB.

        Args:
            xyz: Value (3,) or (B, 3) tensor.

        Returns:
            torch.Tensor: Same shape as input, clamped.
        """
        p = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        return torch.minimum(torch.maximum(p, self.min), self.max)

    def union(self, other: AABB) -> AABB:
        """Compute the union of two AABBs.

        Args:
            other: Another AABB on the same device.

        Returns:
            AABB: Enclosing AABB.
        """
        return AABB(
            min=torch.minimum(self.min, other.min),
            max=torch.maximum(self.max, other.max),
            device=self.device,
        )

    def intersection(self, other: AABB) -> AABB | None:
        """Compute the intersection of two AABBs (returns None if disjoint).

        Args:
            other: Another AABB on the same device.

        Returns:
            AABB: Overlapping region, or None if there is no overlap.
        """
        lo = torch.maximum(self.min, other.min)
        hi = torch.minimum(self.max, other.max)
        if torch.any(lo > hi):
            return None
        return AABB(min=lo, max=hi, device=self.device)

    def half(self, axis: str = "y", side: str = "left") -> AABB:
        """Split along an axis and return one half.

        Args:
            axis: One of {"x", "y", "z"}.
            side: Value "left"/"low" or "right"/"high".

        Returns:
            AABB: The requested half.
        """
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        mid = 0.5 * (self.min[idx] + self.max[idx])
        lo, hi = self.min.clone(), self.max.clone()
        if side in ("left", "low"):
            hi[idx] = mid
        else:
            lo[idx] = mid
        return AABB(min=lo, max=hi, device=self.device)
