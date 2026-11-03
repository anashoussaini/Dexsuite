"""CPU-only checks of the paint-canvas stencil math and the spray_graffiti stencil shapes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dexsuite.utils.paint_canvas as pc
from dexsuite.environments.single.rigid import spray_graffiti as sg

H, W = 60, 80


def _square(r0, r1, c0, c1):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def test_stencil_outline_is_a_ring_of_the_requested_width() -> None:
    interior = _square(20, 40, 30, 50)
    ring = pc.stencil_outline(interior, 5)
    row = ring[30]  # through the middle: 3 px outside + 2 px inside on each vertical edge
    assert row[27:33].tolist() == [True] * 5 + [False]
    assert row[47:53].tolist() == [False] + [True] * 5
    assert not ring[30, 33:47].any() and not ring[30, :27].any() and not ring[30, 53:].any()
    assert not ring[23:37, 33:47].any()  # deep interior untouched
    assert ring[19, 30] and ring[40, 30]  # boundary rows above/below are covered
    ring1 = pc.stencil_outline(interior, 1)
    assert ring1[30, 29] and not ring1[30, 30] and not ring1[30, 28]  # width 1 = one texel outside


def test_region_coverage_semantics() -> None:
    interior = _square(20, 40, 30, 50)
    mask = np.zeros((H, W), dtype=np.float32)
    assert pc.region_coverage(mask, interior, 0.5, None) == 0.0
    assert pc.region_coverage(mask, interior, 0.5, "inside") == 0.0
    assert pc.region_coverage(mask, interior, 0.5, "outside") == 0.0  # nothing painted -> 0, not nan
    mask[20:30, 30:50] = 1.0  # top half of the interior
    assert pc.region_coverage(mask, interior, 0.5, None) == pytest.approx(200 / (H * W))
    assert pc.region_coverage(mask, interior, 0.5, "inside") == pytest.approx(0.5)
    assert pc.region_coverage(mask, interior, 0.5, "outside") == 0.0
    mask[0:10, 0:20] = 1.0  # same mass outside
    assert pc.region_coverage(mask, interior, 0.5, "inside") == pytest.approx(0.5)
    assert pc.region_coverage(mask, interior, 0.5, "outside") == pytest.approx(0.5)
    mask[0:10, 0:20] = 0.25  # mass, not thresholded texels
    assert pc.region_coverage(mask, interior, 0.5, "outside") == pytest.approx(50 / 250)
    with pytest.raises(ValueError):
        pc.region_coverage(mask, None, 0.5, "inside")
    with pytest.raises(ValueError):
        pc.region_coverage(mask, interior, 0.5, "everywhere")


def _fake_gs():
    def entity_factory(*_a, **_k):
        return SimpleNamespace(is_built=False, links=[])

    return SimpleNamespace(
        morphs=SimpleNamespace(Mesh=lambda **k: k),
        materials=SimpleNamespace(Rigid=lambda **k: k),
        surfaces=SimpleNamespace(Default=lambda **k: k),
        textures=SimpleNamespace(ImageTexture=lambda **k: k),
        logger=SimpleNamespace(warning=lambda *_a, **_k: None),
        _entity=entity_factory,
    )


def _canvas(monkeypatch, res):
    fake = _fake_gs()
    monkeypatch.setattr(pc, "gs", fake)
    scene = SimpleNamespace(add_entity=lambda morph, material, surface: fake._entity())
    return pc.PaintCanvas(scene, size=sg.PAPER_SIZE, res=res, base_rgb=(245, 245, 242), paint_rgb=(20, 20, 24))


@pytest.fixture
def canvas(monkeypatch):
    return _canvas(monkeypatch, (H, W))


def test_canvas_without_stencil_composes_the_plain_blend(canvas) -> None:
    canvas.splat(np.array([[0.5, 0.5]]), 0.03, 1.0)
    canvas._compose()
    m = canvas.mask[..., None]
    expected = np.clip(canvas._base * (1.0 - m) + canvas._paint * m + 0.5, 0, 255).astype(np.uint8)
    assert np.array_equal(canvas.image[..., :3], expected)
    assert canvas.stencil is None and canvas.outline is None
    assert canvas.coverage() == pytest.approx(float((canvas.mask > 0.5).mean()))


def test_canvas_stencil_outline_scoring_and_reset(canvas) -> None:
    interior = _square(20, 40, 30, 50)
    canvas.set_stencil(interior, outline_rgb=(220, 30, 30), outline_px=3)
    canvas._compose()
    red = (canvas.image[..., :3] == (220, 30, 30)).all(axis=-1)
    assert np.array_equal(red, canvas.outline)
    assert red.any() and not red[25:35, 33:47].any()
    canvas.mask[22:38, 32:48] = 1.0  # paint the interior only
    assert canvas.coverage(region="inside") == pytest.approx(16 * 16 / 400)
    assert canvas.coverage(region="outside") == 0.0
    canvas.mask[0:5, 0:5] = 1.0
    assert canvas.coverage(region="outside") == pytest.approx(25 / (256 + 25))
    canvas._compose()
    assert (canvas.image[30, 40, :3] == (20, 20, 24)).all()  # paint covers the interior
    canvas.reset()
    assert not canvas.mask.any() and canvas.coverage() == 0.0 and canvas.coverage(region="outside") == 0.0
    canvas._compose()
    assert np.array_equal((canvas.image[..., :3] == (220, 30, 30)).all(axis=-1), red)  # outline survives reset
    canvas.clear_stencil()
    canvas._compose()
    assert (canvas.image[..., :3] == (245, 245, 242)).all()
    with pytest.raises(ValueError):
        canvas.set_stencil(np.zeros((H + 1, W), dtype=bool))
    with pytest.raises(ValueError):
        canvas.coverage(region="inside")


def test_default_outline_width_is_about_4mm_on_the_paper(monkeypatch) -> None:
    paper = _canvas(monkeypatch, sg.PAPER_RES)
    px = int(round(0.004 * paper.px_per_m))
    assert px == 5  # 384 x 512 texels on A3 -> ~1256 px/m
    paper.set_stencil(sg.stencil_mask("square", (0.0, 0.0), 0.15), outline_rgb=sg.STENCIL_RGB, outline_m=sg.STENCIL_LINE_M)
    mid = sg.PAPER_RES[0] // 2
    assert int(paper.outline[mid].sum()) == 2 * px  # left + right edge, 5 texels each
    assert 3.5e-3 <= px / paper.px_per_m <= 4.5e-3


def _mask_bbox_m(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(mask)
    Hp, Wp = sg.PAPER_RES
    x0 = cols.min() / Wp * sg.PAPER_SIZE[0] - 0.5 * sg.PAPER_SIZE[0]
    x1 = (cols.max() + 1) / Wp * sg.PAPER_SIZE[0] - 0.5 * sg.PAPER_SIZE[0]
    y1 = 0.5 * sg.PAPER_SIZE[1] - rows.min() / Hp * sg.PAPER_SIZE[1]
    y0 = 0.5 * sg.PAPER_SIZE[1] - (rows.max() + 1) / Hp * sg.PAPER_SIZE[1]
    return x0, x1, y0, y1


@pytest.mark.parametrize("shape", sg.STENCIL_SHAPES)
def test_stencil_shapes_have_the_requested_size_and_area(shape) -> None:
    size = 0.15
    mask = sg.stencil_mask(shape, (0.02, -0.01), size)
    assert mask.shape == tuple(sg.PAPER_RES)
    x0, x1, y0, y1 = _mask_bbox_m(mask)
    hx, hy = sg.stencil_half_extents(shape, size)
    assert (x1 - x0) == pytest.approx(2 * hx, abs=2.5e-3) and (y1 - y0) == pytest.approx(2 * hy, abs=2.5e-3)
    assert 0.5 * (x0 + x1) == pytest.approx(0.02, abs=2e-3) and 0.5 * (y0 + y1) == pytest.approx(-0.01, abs=2e-3)
    texel_area = (sg.PAPER_SIZE[0] / sg.PAPER_RES[1]) * (sg.PAPER_SIZE[1] / sg.PAPER_RES[0])
    area = mask.sum() * texel_area
    expected = {
        "circle": math.pi * (0.5 * size) ** 2,
        "square": size**2,
        "triangle": 0.25 * math.sqrt(3.0) * size**2,
        "letter_f": size**2 * (sg.F_STROKE + (sg.F_WIDTH - sg.F_STROKE) * sg.F_STROKE + (0.75 * sg.F_WIDTH - sg.F_STROKE) * sg.F_STROKE),
    }[shape]
    assert area == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("shape", sg.STENCIL_SHAPES)
@pytest.mark.parametrize("u", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.37, 0.5, 0.5)])
def test_stencil_pose_keeps_the_margin_and_size_range(shape, u) -> None:
    center, size = sg.stencil_pose(shape, torch.tensor(u))
    assert sg.STENCIL_SIZE_RANGE[0] <= size <= sg.STENCIL_SIZE_RANGE[1]
    x0, x1, y0, y1 = _mask_bbox_m(sg.stencil_mask(shape, center, size))
    tol = 1.5e-3  # one texel
    assert x0 >= -0.5 * sg.PAPER_SIZE[0] + sg.STENCIL_MARGIN - tol and x1 <= 0.5 * sg.PAPER_SIZE[0] - sg.STENCIL_MARGIN + tol
    assert y0 >= -0.5 * sg.PAPER_SIZE[1] + sg.STENCIL_MARGIN - tol and y1 <= 0.5 * sg.PAPER_SIZE[1] - sg.STENCIL_MARGIN + tol
