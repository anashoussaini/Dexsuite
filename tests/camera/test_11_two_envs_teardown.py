# tests/camera/test_11_two_envs_teardown.py
"""Freeing one env (close() or the garbage collector) must leave a second env's cameras working and unchanged."""

from __future__ import annotations

import gc

import pytest

import dexsuite as ds


def _make():
    return ds.make(
        "lift",
        manipulator="franka",
        gripper="robotiq",
        cameras=["front"],
        modalities=["rgb", "depth"],
    )


@pytest.mark.parametrize("free", ["close", "gc"])
def test_freeing_one_env_keeps_the_other_envs_cameras(free):
    first = _make()
    first.reset(seed=1)
    second = _make()
    before, _ = second.reset(seed=7)
    before = before["cameras"]["front"]["rgb"].clone()
    if free == "close":
        first.close()
        first.close()  # a second close is a no-op
    del first
    gc.collect()
    after, _ = second.reset(seed=7)
    second.step(second.action_space.sample())
    assert (after["cameras"]["front"]["rgb"] == before).all()
    second.close()
