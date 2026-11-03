"""Write the 1 x 1 m UV-mapped slab used by ``dexsuite.utils.paint_canvas.PaintCanvas``.

A 1 x 1 x 0.001 m closed box whose +Z face (z = +0.0005) is the paintable canvas: UV (0,0) at its
(-x, -y) corner and (1,1) at (+x, +y), so ``u = x + 0.5`` and ``v = y + 0.5``. The other faces map to
UV (0,0). A closed volume is required because Genesis derives a mass from visual geometry when the
canvas is attached to a moving link.

Run from the repository root::

    python scripts/assets/make_paint_quad.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "dexsuite" / "models" / "objects" / "task" / "paint_quad" / "paint_quad.obj"

OBJ = """# paint_quad: 1 x 1 x 0.001 m slab, +Z face is the canvas with UV = (x + 0.5, y + 0.5)
v -0.5 -0.5 0.0005
v  0.5 -0.5 0.0005
v  0.5  0.5 0.0005
v -0.5  0.5 0.0005
v -0.5 -0.5 -0.0005
v  0.5 -0.5 -0.0005
v  0.5  0.5 -0.0005
v -0.5  0.5 -0.0005
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
vn 0.0 -1.0 0.0
vn 1.0 0.0 0.0
vn 0.0 1.0 0.0
vn -1.0 0.0 0.0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
f 5/1/2 7/1/2 6/1/2
f 5/1/2 8/1/2 7/1/2
f 1/1/3 6/1/3 2/1/3
f 1/1/3 5/1/3 6/1/3
f 2/1/4 7/1/4 3/1/4
f 2/1/4 6/1/4 7/1/4
f 3/1/5 8/1/5 4/1/5
f 3/1/5 7/1/5 8/1/5
f 4/1/6 5/1/6 1/1/6
f 4/1/6 8/1/6 5/1/6
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(OBJ)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
