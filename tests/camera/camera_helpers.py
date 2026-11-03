"""Shared assertions for camera observation tests."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def is_integer_dtype(arr) -> bool:
    return np.issubdtype(to_numpy(arr).dtype, np.integer)


def is_float_dtype(arr) -> bool:
    return np.issubdtype(to_numpy(arr).dtype, np.floating)


def assert_modalities_exact(pack: dict, expected: tuple[str, ...]):
    assert set(pack.keys()) == set(expected), (
        f"modalities mismatch: {pack.keys()} vs {expected}"
    )


def assert_rgb_shape(arr, height: int | None = None, width: int | None = None):
    image = to_numpy(arr)
    assert image.ndim == 3 and image.shape[-1] == 3, (
        f"RGB must be HxWx3, got {image.shape}"
    )
    if height is not None:
        assert image.shape[0] == height, (
            f"RGB H mismatch: got {image.shape[0]}, expected {height}"
        )
    if width is not None:
        assert image.shape[1] == width, (
            f"RGB W mismatch: got {image.shape[1]}, expected {width}"
        )
    assert image.dtype == np.uint8, f"RGB dtype must be uint8; got {image.dtype}"


def assert_depth_shape(arr, height: int | None = None, width: int | None = None):
    depth = to_numpy(arr)
    assert depth.ndim == 2, f"Depth must be HxW, got {depth.shape}"
    if height is not None:
        assert depth.shape[0] == height, (
            f"Depth H mismatch: got {depth.shape[0]}, expected {height}"
        )
    if width is not None:
        assert depth.shape[1] == width, (
            f"Depth W mismatch: got {depth.shape[1]}, expected {width}"
        )
    assert is_float_dtype(depth), f"Depth must be float dtype; got {depth.dtype}"


def assert_seg_shape(arr, height: int | None = None, width: int | None = None):
    segmentation = to_numpy(arr)
    assert segmentation.ndim == 2, (
        f"Segmentation must be HxW, got {segmentation.shape}"
    )
    if height is not None:
        assert segmentation.shape[0] == height, (
            f"Seg H mismatch: got {segmentation.shape[0]}, expected {height}"
        )
    if width is not None:
        assert segmentation.shape[1] == width, (
            f"Seg W mismatch: got {segmentation.shape[1]}, expected {width}"
        )
    assert is_integer_dtype(segmentation), (
        f"Segmentation must be integer dtype; got {segmentation.dtype}"
    )


def assert_normal_shape(arr, height: int | None = None, width: int | None = None):
    normal = to_numpy(arr)
    assert normal.ndim == 3 and normal.shape[-1] == 3, (
        f"Normal must be HxWx3, got {normal.shape}"
    )
    if height is not None:
        assert normal.shape[0] == height, (
            f"Normal H mismatch: got {normal.shape[0]}, expected {height}"
        )
    if width is not None:
        assert normal.shape[1] == width, (
            f"Normal W mismatch: got {normal.shape[1]}, expected {width}"
        )
    assert is_integer_dtype(normal) or is_float_dtype(normal), (
        f"Normal must be numeric; got {normal.dtype}"
    )


def assert_pack_shapes(
    pack: dict,
    *,
    height: int | None = None,
    width: int | None = None,
):
    if "rgb" in pack:
        assert_rgb_shape(pack["rgb"], height, width)
    if "depth" in pack:
        assert_depth_shape(pack["depth"], height, width)
    if "segmentation" in pack:
        assert_seg_shape(pack["segmentation"], height, width)
    if "normal" in pack:
        assert_normal_shape(pack["normal"], height, width)
