import numpy as np
import pytest
import torch
from gymnasium import spaces

# Under test
import dexsuite.utils.aabb as aabb_mod
import dexsuite.utils.randomizers as rnd
from dexsuite.utils.space_utils import dict_to_space
from dexsuite.utils.workspace_utils import outside_aabb

# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------


def _patch_device_and_envs(monkeypatch, *, device="cpu", n_envs=1):
    """Force the utils to use a known device and n_envs.

    We patch the imported symbols in the target modules (not the globals provider),
    because the modules under test captured the names on import.
    """
    dev = torch.device(device)
    monkeypatch.setattr(aabb_mod, "get_device", lambda: dev, raising=True)
    monkeypatch.setattr(rnd, "get_device", lambda: dev, raising=True)
    monkeypatch.setattr(rnd, "get_n_envs", lambda: int(n_envs), raising=True)


def _mk_box():
    # Franka example from your workspace table
    return aabb_mod.AABB.from_lists([0.10, -0.40, -0.10], [0.65, 0.40, 0.40])


def _all_inside(box, points):
    return bool(box.contains(points).all())


# ---------------------------------------------------------
# AABB basics
# ---------------------------------------------------------


def test_aabb_contains_and_clamp_shapes(monkeypatch):
    _patch_device_and_envs(monkeypatch, device="cpu", n_envs=1)
    box = _mk_box()

    # 1D point
    p = torch.tensor([0.5, 0.0, 0.2], device=box.device)
    inside_1d = box.contains(p)
    assert inside_1d.shape == (1,) and inside_1d.dtype == torch.bool
    assert bool(inside_1d.item()) is True
    # (B, 3) batch
    points = torch.tensor([[0.5, 0.0, 0.2], [2.0, 0.0, 0.2]], device=box.device)
    inside = box.contains(points)
    assert inside.shape == (2,) and inside.dtype == torch.bool
    assert inside.tolist() == [True, False]

    # clamp preserves shape
    clamped_points = box.clamp(points)
    assert clamped_points.shape == (2, 3)
    assert _all_inside(box, clamped_points)


def test_aabb_center_and_margin(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=1)
    box = _mk_box()
    c = box.center()
    assert c.shape == (3,)
    # with_margin never inverts
    shrink = box.with_margin(0.10)
    assert torch.all(shrink.min <= shrink.max)
    # center unaffected by symmetric margins
    assert torch.allclose(shrink.center(), c)


# ---------------------------------------------------------
# sample_in_aabb_uniform (delegates to ThreeDPosRandomizer)
# ---------------------------------------------------------


def test_uniform_single_env_returns_1d_inside(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=1)
    box = _mk_box()
    torch.manual_seed(0)
    p = rnd.sample_in_aabb_uniform(box)  # no env_idx, n_envs==1 -> (3,)
    assert p.shape == (3,)
    assert _all_inside(box, p)


def test_uniform_vectorized_returns_2d_inside(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=16)
    box = _mk_box()
    torch.manual_seed(0)
    points = rnd.sample_in_aabb_uniform(box)  # env_idx None, n_envs>1 -> (B,3)
    assert points.ndim == 2 and points.shape[0] == 16 and points.shape[1] == 3
    assert _all_inside(box, points)


def test_uniform_subset_env_idx_controls_batch_and_inside(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=8)
    box = _mk_box()
    torch.manual_seed(0)
    # list env_idx
    points = rnd.sample_in_aabb_uniform(box, env_idx=[0, 3, 5])
    assert points.shape == (3, 3) and _all_inside(box, points)
    # tensor env_idx of length 1 -> still 2D (1,3)
    one_point = rnd.sample_in_aabb_uniform(box, env_idx=torch.tensor([7]))
    assert one_point.shape == (1, 3) and _all_inside(box, one_point)


def test_uniform_with_margin_inside_shrunk_box(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=4)
    box = _mk_box()
    m = 0.05
    points = rnd.sample_in_aabb_uniform(box, env_idx=[0, 1, 2, 3], margin=m)
    # points should be inside the margin-shrunk box
    shrunk = box.with_margin(m)
    assert _all_inside(shrunk, points)


def test_uniform_extreme_margin_collapses_to_center(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=4)
    box = _mk_box()
    points = rnd.sample_in_aabb_uniform(
        box,
        env_idx=[0, 1, 2, 3],
        margin=999.0,
    )
    center = box.center().reshape(1, 3).expand_as(points)
    assert torch.allclose(points, center)


# ---------------------------------------------------------
# sample_in_aabb_center
# ---------------------------------------------------------


def test_center_single_vs_subset(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=1)
    box = _mk_box()
    p = rnd.sample_in_aabb_center(box)  # (3,)
    assert p.shape == (3,)
    assert torch.allclose(p, box.center())

    points = rnd.sample_in_aabb_center(box, env_idx=[2, 4])  # (2,3)
    assert points.shape == (2, 3)
    assert torch.allclose(points[0], box.center()) and torch.allclose(
        points[1],
        box.center(),
    )


# ---------------------------------------------------------
# sample_in_aabb_center_xy_band
# ---------------------------------------------------------


def test_center_xy_band_square_inside_and_shape(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=6)
    box = _mk_box()
    torch.manual_seed(1)
    band = (0.10, 0.05)
    points = rnd.sample_in_aabb_center_xy_band(
        box,
        band_xy=band,
        mode="square",
        z="center",
        env_idx=[0, 1, 2, 3, 4, 5],
    )
    assert points.shape == (6, 3) and _all_inside(box, points)
    # offsets from center should be within band (allow 1e-6 for clamp)
    c = box.center()
    dxy = points[:, :2] - c[:2]
    assert torch.all(torch.abs(dxy[:, 0]) <= band[0] + 1e-6)
    assert torch.all(torch.abs(dxy[:, 1]) <= band[1] + 1e-6)
    # z fixed to center
    assert torch.allclose(points[:, 2], c[2].expand_as(points[:, 2]))


def test_center_xy_band_circle_inside_ellipse_and_uniform_z(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=5)
    box = _mk_box()
    torch.manual_seed(2)
    band = (0.10, 0.05)
    points = rnd.sample_in_aabb_center_xy_band(
        box,
        band_xy=band,
        mode="circle",
        z="uniform",
        env_idx=[0, 1, 2, 3, 4],
    )
    assert points.shape == (5, 3) and _all_inside(box, points)
    c = box.center()
    dx = (points[:, 0] - c[0]) / band[0]
    dy = (points[:, 1] - c[1]) / band[1]
    # inside ellipse: (dx/bx)^2 + (dy/by)^2 <= 1 (allow small slack)
    assert torch.all(dx**2 + dy**2 <= 1.0000001)
    # z must lie within box z-extent
    z_lo, z_hi = float(box.min[2]), float(box.max[2])
    assert torch.all(
        (points[:, 2] >= z_lo - 1e-6) & (points[:, 2] <= z_hi + 1e-6),
    )


def test_center_xy_band_extreme_margin_collapses_to_center(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=3)
    box = _mk_box()
    points = rnd.sample_in_aabb_center_xy_band(
        box,
        band_xy=1.0,
        env_idx=[0, 1, 2],
        margin=999.0,
    )
    center = box.center().reshape(1, 3).expand_as(points)
    assert torch.allclose(points, center)


# ---------------------------------------------------------
# sample_around_point_xy
# ---------------------------------------------------------


def test_around_point_xy_clamped_and_z_fixed(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=3)
    box = _mk_box()
    torch.manual_seed(3)
    anchor_xy = [10.0, 10.0]  # deliberately outside -> clamp should bring inside
    z_fixed = 0.20
    points = rnd.sample_around_point_xy(
        anchor_xy,
        band_xy=0.15,
        z=z_fixed,
        mode="circle",
        env_idx=[0, 1, 2],
        clamp_to=box,
    )
    assert points.shape == (3, 3) and _all_inside(box, points)
    assert torch.allclose(points[:, 2], torch.full((3,), z_fixed))


# ---------------------------------------------------------
# TwoDPosRandomizer & ThreeDPosRandomizer direct behavior
# ---------------------------------------------------------


def test_twod_randomizer_shapes_and_clamp(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=1)
    box = _mk_box()
    r2d = rnd.TwoDPosRandomizer((0.1, 0.65), (-0.4, 0.4), z=0.2)

    torch.manual_seed(4)
    p = r2d.sample()  # n_envs==1, no env_idx -> (3,)
    assert p.shape == (3,)
    points = r2d.sample(env_idx=[0, 1, 2], clamp_to=box)  # (B,3)
    assert points.shape == (3, 3) and _all_inside(box, points)

    # circle band around center of box
    torch.manual_seed(5)
    cxy = box.center()[:2]
    circle_points = r2d.sample(
        env_idx=[0, 1],
        around_xy=cxy,
        band_xy=(0.05, 0.05),
        mode="circle",
    )
    assert circle_points.shape == (2, 3)


def test_threed_randomizer_freeze_axes(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=5)
    r3d = rnd.ThreeDPosRandomizer(
        x_range=(0.1, 0.65),
        y_range=(-0.4, 0.4),
        z_range=(0.0, 0.4),
    )
    torch.manual_seed(6)
    points = r3d.sample(
        env_idx=[0, 1, 2, 3, 4],
        freeze_axes=("z",),
        freeze_values=[0.0, 0.0, 0.20],
    )
    assert points.shape == (5, 3)
    assert torch.allclose(points[:, 2], torch.full((5,), 0.20))


# ---------------------------------------------------------
# sampler_in_aabb wrapper
# ---------------------------------------------------------


def test_sampler_in_aabb_clamp_policy(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=3)
    box = _mk_box()

    # Base sampler that returns way out-of-bounds
    def base_sampler(*, env_idx=None):
        batch_size = 1 if env_idx is None else len(env_idx)
        return torch.full((batch_size, 3), -9999.0)

    safe = rnd.sampler_in_aabb(base_sampler, box, policy="clamp")
    points = safe(env_idx=[0, 1, 2])
    assert points.shape == (3, 3) and _all_inside(box, points)


def test_sampler_in_aabb_rejection_then_clamp(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=2)
    box = _mk_box()

    # Always outside; set max_tries=0 to force immediate clamp
    def base_sampler(*, env_idx=None):
        batch_size = 1 if env_idx is None else len(env_idx)
        return torch.full((batch_size, 3), 1e6)

    safe = rnd.sampler_in_aabb(base_sampler, box, policy="rejection", max_tries=0)
    points = safe(env_idx=[5, 6])
    assert points.shape == (2, 3) and _all_inside(box, points)


# ---------------------------------------------------------
# dict_to_space convenience
# ---------------------------------------------------------


def test_dict_to_space_basic(monkeypatch):
    _patch_device_and_envs(monkeypatch, n_envs=1)
    leaf_t = torch.zeros(2, 3)
    tree = {"a": leaf_t, "b": {"c": np.zeros((4,), dtype=np.float32), "d": 1.0}}
    space = dict_to_space(tree)
    assert isinstance(space, spaces.Dict)
    assert tuple(space["a"].shape) == (2, 3)
    assert tuple(space["b"]["c"].shape) == (4,)
    assert tuple(space["b"]["d"].shape) == ()


def test_outside_aabb_accepts_tensor_bounds():
    pos = torch.tensor(
        [
            [0.5, 0.0, 0.2],
            [2.0, 0.0, 0.2],
            [0.5, -1.0, 0.2],
        ],
    )
    lo = torch.tensor([0.1, -0.4, -0.1])
    hi = torch.tensor([0.65, 0.4, 0.4])
    assert outside_aabb(pos, lo, hi).tolist() == [False, True, True]


# ---------------------------------------------------------
# Shape discipline across APIs
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "n_envs,env_idx,expected_ndim,expected_batch_size",
    [
        (1, None, 1, None),  # single env, no subset -> (3,)
        (8, None, 2, 8),  # vectorized, full batch
        (8, [2], 2, 1),  # subset length 1 -> (1,3)
        (8, [2, 4, 6], 2, 3),  # subset length 3 -> (3,3)
    ],
)
def test_shape_rules_uniform(
    monkeypatch,
    n_envs,
    env_idx,
    expected_ndim,
    expected_batch_size,
):
    _patch_device_and_envs(monkeypatch, n_envs=n_envs)
    box = _mk_box()
    points = rnd.sample_in_aabb_uniform(box, env_idx=env_idx)
    assert points.ndim == expected_ndim
    if expected_ndim == 1:
        assert points.shape == (3,)
    else:
        assert points.shape == (expected_batch_size, 3)
    assert _all_inside(box, points)
