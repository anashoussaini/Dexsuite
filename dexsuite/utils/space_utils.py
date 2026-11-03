"""Build Gymnasium spaces for torch-backed observations and actions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from gymnasium import spaces


def _box_value(value: Any, dtype: type[np.generic], shape: tuple[int, ...] | None) -> Any:
    if shape is not None and isinstance(value, int | float):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _make_box(
    low: Any,
    high: Any,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: type[np.generic] = np.float32,
) -> spaces.Box:
    """Create a Gymnasium Box while accepting Torch tensors."""
    return spaces.Box(
        low=_box_value(low, dtype, shape),
        high=_box_value(high, dtype, shape),
        shape=shape,
        dtype=dtype,
    )


def float_box(
    low: Any,
    high: Any,
    *,
    shape: tuple[int, ...] | None = None,
) -> spaces.Box:
    """Create a float32 Gymnasium Box from scalar, sequence, or Torch bounds.

    Args:
        low: Lower bound for the created space.
        high: Upper bound for the created space.
        shape: Shape for the created space. Defaults to None.
    """
    return _make_box(low, high, shape=shape, dtype=np.float32)


def uint8_box(low: int, high: int, *, shape: tuple[int, ...]) -> spaces.Box:
    """Create a uint8 image Gymnasium Box.

    Args:
        low: Lower bound for the created space.
        high: Upper bound for the created space.
        shape: Shape for the created space.
    """
    return _make_box(low, high, shape=shape, dtype=np.uint8)


def int32_box(low: int, high: int, *, shape: tuple[int, ...]) -> spaces.Box:
    """Create an int32 Gymnasium Box.

    Args:
        low: Lower bound for the created space.
        high: Upper bound for the created space.
        shape: Shape for the created space.
    """
    return _make_box(low, high, shape=shape, dtype=np.int32)


def dict_to_space(tree: Any, bound: float = math.inf) -> spaces.Space:
    """Recursively convert a nested observation prototype to Gymnasium spaces.

    Args:
        tree: Nested observation prototype to convert.
        bound: Finite fallback bound for floating values. Defaults to math.inf.
    """
    if isinstance(tree, dict):
        return spaces.Dict({key: dict_to_space(value, bound) for key, value in tree.items()})
    if torch.is_tensor(tree):
        return float_box(-bound, bound, shape=tuple(tree.shape))

    tensor = torch.as_tensor(tree, dtype=torch.float32)
    return float_box(-bound, bound, shape=tuple(tensor.shape))


def box_numel(space: spaces.Box) -> int:
    """Return the flattened number of values in a Gymnasium Box.

    Args:
        space: Gymnasium box space to inspect.
    """
    return int(math.prod(space.shape))


def _flat_box_bounds(space: spaces.Box) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened float32 low/high tensors for a Gymnasium Box."""
    low = torch.as_tensor(space.low, dtype=torch.float32).reshape(-1)
    high = torch.as_tensor(space.high, dtype=torch.float32).reshape(-1)
    return low, high


def concat_box_bounds(boxes: Iterable[spaces.Box]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate flattened bounds from multiple Gymnasium boxes.

    Args:
        boxes: Box spaces whose bounds should be concatenated.
    """
    lows: list[torch.Tensor] = []
    highs: list[torch.Tensor] = []
    for box in boxes:
        low, high = _flat_box_bounds(box)
        lows.append(low)
        highs.append(high)
    return torch.cat(lows, dim=0), torch.cat(highs, dim=0)


def frame_to_tensor(frame: Any) -> torch.Tensor:
    """Convert a rendered frame to a contiguous Torch tensor.

    Args:
        frame: Rendered frame to convert.
    """
    if torch.is_tensor(frame):
        return frame.contiguous()
    if isinstance(frame, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(frame))
    return torch.as_tensor(frame).contiguous()
