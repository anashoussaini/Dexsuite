# tests/robot/test_01_aabb_defaults_and_validation.py
import pytest

from dexsuite.options.robot import AABBOptions


def test_aabb_defaults_are_valid_from_yaml():
    aabb = AABBOptions()
    lo, hi = aabb.min, aabb.max
    assert len(lo) == 3 and len(hi) == 3
    assert all(lo[i] <= hi[i] for i in range(3))


def test_aabb_raises_on_swapped_bounds():
    with pytest.raises(ValueError):
        _ = AABBOptions(min=(1.0, 0.0, 0.0), max=(0.0, 1.0, 1.0))
