"""Action layout and dispatch helpers for robot controllers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from gymnasium import spaces

from dexsuite.utils.space_utils import box_numel, concat_box_bounds, float_box


@dataclass(frozen=True)
class _ActionSegment:
    """Contiguous section of a flat action vector."""

    start: int
    stop: int
    ctrl: Any
    name: tuple[str, ...]

    def width(self) -> int:
        """Return the dimensionality of this action segment."""
        return int(self.stop - self.start)


@dataclass
class ActionLayout:
    """Validated layout for a flat, composite action vector."""

    segments: tuple[_ActionSegment, ...]
    total_dim: int

    def __post_init__(self) -> None:
        """Validate segment structure and cache flat names."""
        self._validate_contiguity()
        self._validate_dims()
        self._flat_keys = tuple(".".join(seg.name) for seg in self.segments)

    def as_box(self) -> spaces.Box:
        """Return the concatenated Gymnasium action space."""
        boxes: list[spaces.Box] = []
        for seg in self.segments:
            sp = seg.ctrl.action_space()
            if not isinstance(sp, spaces.Box):
                raise TypeError(
                    f"Controller {type(seg.ctrl).__name__} must return spaces.Box",
                )
            boxes.append(sp)
        low_cat, high_cat = concat_box_bounds(boxes)

        return float_box(low_cat, high_cat, shape=(self.total_dim,))

    def flat_keys(self) -> tuple[str, ...]:
        """Return stable dotted names for each segment."""
        return self._flat_keys

    def views(self, action_2d: torch.Tensor) -> Iterable[torch.Tensor]:
        """Yield zero-copy per-segment views into a flat action tensor.

        Args:
            action_2d: Flat batched action tensor.
        """
        if action_2d.ndim == 1:
            for seg in self.segments:
                yield action_2d[seg.start : seg.stop]
        elif action_2d.ndim == 2:
            for seg in self.segments:
                yield action_2d[:, seg.start : seg.stop]
        else:
            raise ValueError(f"Action must be 1D or 2D; got {action_2d.ndim}D")

    def _validate_contiguity(self) -> None:
        """Check that segments exactly fill the flat action dimension."""
        cur = 0
        for seg in self.segments:
            if seg.start != cur:
                raise ValueError(
                    f"Non-contiguous layout: expected start {cur}, got {seg.start}",
                )
            if seg.stop <= seg.start:
                raise ValueError(
                    f"Empty/invalid segment {seg.name}: [{seg.start},{seg.stop})",
                )
            cur = seg.stop
        if cur != self.total_dim:
            raise ValueError(f"Layout covers 0..{cur}, but total_dim={self.total_dim}")

    def _validate_dims(self) -> None:
        """Check segment widths against controller action spaces."""
        for seg in self.segments:
            sp = seg.ctrl.action_space()
            need = seg.width()
            got = box_numel(sp)
            if need != got:
                raise ValueError(
                    f"Segment {seg.name}: width {need} != controller action dim {got}",
                )


def _dims(ctrl: Any) -> int:
    """Return the flattened controller action-space dimension."""
    sp = ctrl.action_space()
    return box_numel(sp)


def build_single_layout(*, arm_ctrl: Any, grip_ctrl: Any | None) -> ActionLayout:
    """Build a single-arm layout: manipulator, then optional gripper.

    Args:
        arm_ctrl: Manipulator controller for a single-arm robot.
        grip_ctrl: Gripper controller for a single-arm robot.
    """
    segs: list[_ActionSegment] = []
    cur = 0

    w = _dims(arm_ctrl)
    segs.append(
        _ActionSegment(start=cur, stop=cur + w, ctrl=arm_ctrl, name=("manipulator",)),
    )
    cur += w

    if grip_ctrl is not None:
        w = _dims(grip_ctrl)
        segs.append(
            _ActionSegment(start=cur, stop=cur + w, ctrl=grip_ctrl, name=("gripper",)),
        )
        cur += w

    return ActionLayout(segments=tuple(segs), total_dim=cur)


def build_bimanual_layout(
    *,
    left_arm: Any,
    left_grip: Any | None,
    right_arm: Any,
    right_grip: Any | None,
) -> ActionLayout:
    """Build a bimanual layout: left controls first, then right controls.

    Args:
        left_arm: Left manipulator controller.
        left_grip: Left gripper controller.
        right_arm: Right manipulator controller.
        right_grip: Right gripper controller.
    """
    segs: list[_ActionSegment] = []
    cur = 0

    wl = _dims(left_arm)
    segs.append(
        _ActionSegment(
            start=cur,
            stop=cur + wl,
            ctrl=left_arm,
            name=("left", "manipulator"),
        ),
    )
    cur += wl

    if left_grip is not None:
        wlg = _dims(left_grip)
        segs.append(
            _ActionSegment(
                start=cur,
                stop=cur + wlg,
                ctrl=left_grip,
                name=("left", "gripper"),
            ),
        )
        cur += wlg

    wr = _dims(right_arm)
    segs.append(
        _ActionSegment(
            start=cur,
            stop=cur + wr,
            ctrl=right_arm,
            name=("right", "manipulator"),
        ),
    )
    cur += wr

    if right_grip is not None:
        wrg = _dims(right_grip)
        segs.append(
            _ActionSegment(
                start=cur,
                stop=cur + wrg,
                ctrl=right_grip,
                name=("right", "gripper"),
            ),
        )
        cur += wrg

    return ActionLayout(segments=tuple(segs), total_dim=cur)


@torch.no_grad()
def dispatch(layout: ActionLayout, action: torch.Tensor) -> None:
    """Forward flat action slices to their owning controllers.

    Args:
        layout: Action layout used to split the flat action tensor.
        action: Action tensor to validate or dispatch.
    """
    offs = 0
    for seg in layout.segments:
        view = action[..., offs : seg.stop]
        seg.ctrl.step(view)
        offs = seg.stop
