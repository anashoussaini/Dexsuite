"""Rubber duck (Genesis assets/meshes/duck, textured) -> task/duck, the object of pick_place_duck.

Genesis' textured rubber duck (genesis/assets/meshes/duck/duck.obj, Y-up, beak +Z, file units) ->
task/duck: duck.xml + duck/{textured.obj,textured.mtl,texture_map.png,collision_K.stl} + ATTRIBUTION.txt + IMPORT_REPORT.txt.
Duck frame: x = beak direction, y = duck's left, z = up; origin = COM of the solid collision union.
Collision: 5 convex hulls cut from the mesh's own shells (head + eyes, beak, left wing, right wing, body; no CoACD),
each quadric-decimated to <= MAX_F faces and re-hulled, so Genesis neither decimates nor decomposes them.
Deterministic. Usage (from the Dexsuite root; on the shared node cap the BLAS threads):
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I scripts/assets/make_duck.py [out_dir] [length_m] [max_hull_faces]
"""
import hashlib, os, shutil, sys
from pathlib import Path
import numpy as np, trimesh
import genesis as gs

SRC = os.path.join(gs.utils.get_assets_dir(), "meshes", "duck")
REPO = Path(__file__).resolve().parents[2]
OUT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "dexsuite" / "models" / "objects" / "task" / "duck")
LENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.090      # beak tip -> tail tip
MAX_F = int(sys.argv[3]) if len(sys.argv) > 3 else 300            # faces per collision hull (Genesis decimates > 500)
MASS = 0.040
FRICTION = 1.0
NECK_YF, OVERLAP_YF = 88.0, 8.0                                     # file units (Y up): head above, body below (+8 overlap)
P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)             # duck x = file Z, duck y = file X, duck z = file Y
HEAD_LOCAL = (0.010, 0.0, 0.0214)                                   # = pick_place_duck.HEAD_LOCAL (report only)
PINCH_BAND_Z = (0.016, 0.026)                                       # cheek pinch band, aim HEAD_LOCAL z (report only)

raw = trimesh.load(f"{SRC}/duck.obj", force="mesh", process=False)
w = trimesh.Trimesh(np.asarray(raw.vertices), np.asarray(raw.faces), process=True)
w.merge_vertices(merge_tex=True, merge_norm=True)
body_shell, beak, wing_a, wing_b, eye_a, eye_b = sorted(w.split(only_watertight=False), key=lambda c: -len(c.faces))
S = LENGTH / w.extents[2]
to_duck = lambda v: (np.asarray(v) @ P.T) * S
V0 = body_shell.vertices
parts = {
    "head": np.vstack([V0[V0[:, 1] >= NECK_YF], eye_a.vertices, eye_b.vertices]),
    "beak": beak.vertices,
    "wing_left": wing_a.vertices if wing_a.vertices[:, 0].mean() > 0 else wing_b.vertices,   # file +X = duck +y (left)
    "wing_right": wing_b.vertices if wing_a.vertices[:, 0].mean() > 0 else wing_a.vertices,
    "body": V0[V0[:, 1] <= NECK_YF + OVERLAP_YF],
}

import fast_simplification
def simplify_hull(pts, max_f):
    """Quadric-decimate the hull to <= max_f faces (fast_simplification, agg 2), then re-hull."""
    h = trimesh.convex.convex_hull(pts)
    if len(h.faces) <= max_f:
        return h
    v, f = fast_simplification.simplify(h.vertices.astype(np.float32), h.faces.astype(np.int64),
                                        target_reduction=1 - max_f / len(h.faces), agg=2)
    return trimesh.convex.convex_hull(v.astype(float))

hulls = {k: simplify_hull(to_duck(v), MAX_F) for k, v in parts.items()}
full = {k: trimesh.convex.convex_hull(to_duck(v)) for k, v in parts.items()}
loss = {}
for k in hulls:   # deviation of the simplified hull from the full hull (surface samples of the full hull)
    pq = trimesh.proximity.ProximityQuery(hulls[k])
    dd = pq.signed_distance(trimesh.sample.sample_surface(full[k], 20000, seed=1)[0])  # >0: full surface inside simplified
    loss[k] = dict(inward_mm=round(float(np.clip(-dd, 0, None).max() * 1e3), 2), outward_mm=round(float(np.clip(dd, 0, None).max() * 1e3), 2))

# solid union on a 0.5 mm grid -> volume, COM, inertia (uniform density scaled to MASS)
lo = np.min([h.bounds[0] for h in hulls.values()], 0); hi = np.max([h.bounds[1] for h in hulls.values()], 0)
g = 0.0005
axes = [np.arange(lo[i] + g / 2, hi[i], g) for i in range(3)]
X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
inside = np.zeros(len(X), bool)
for h in hulls.values():
    n = h.face_normals; off = np.einsum("ij,ij->i", n, h.triangles[:, 0])
    inside |= np.all(X @ n.T <= off + 1e-12, axis=1)
Pin = X[inside]; vol = len(Pin) * g ** 3; com = Pin.mean(0); r = Pin - com
I = MASS / len(Pin) * (np.eye(3) * np.sum(r * r) - r.T @ r)

os.makedirs(os.path.join(OUT, "duck"), exist_ok=True)
for i, (k, h) in enumerate(hulls.items()):
    hh = h.copy(); hh.apply_translation(-com); hh.export(os.path.join(OUT, "duck", f"collision_{i}.stl"))
# visual: rewrite only the vertex lines (UVs / faces untouched)
lines = open(f"{SRC}/duck.obj").read().splitlines()
outl = ["# duck (Genesis assets/meshes/duck/duck.obj) in the duck frame, metres, origin at the COM"]
for ln in lines:
    if ln.startswith("mtllib"):
        outl.append("mtllib textured.mtl")
    elif ln.startswith("v "):
        v = np.array([float(t) for t in ln.split()[1:4]])
        x, y, z = to_duck(v) - com
        outl.append(f"v {x:.8f} {y:.8f} {z:.8f}")
    elif ln.startswith("#"):
        continue
    else:
        outl.append(ln)
open(os.path.join(OUT, "duck", "textured.obj"), "w").write("\n".join(outl) + "\n")
mtl = open(f"{SRC}/material.mtl").read().replace("material_0.png", "texture_map.png")
open(os.path.join(OUT, "duck", "textured.mtl"), "w").write(mtl)
shutil.copyfile(f"{SRC}/material_0.png", os.path.join(OUT, "duck", "texture_map.png"))

col = "\n".join(f'    <mesh name="duck_collision_mesh_{i}" file="duck/collision_{i}.stl" />' for i in range(len(hulls)))
geoms = "\n".join(f'      <geom name="duck_collision_geom_{i}" type="mesh" mesh="duck_collision_mesh_{i}" contype="1" conaffinity="1" group="3" friction="{FRICTION} 0.005 0.0001" />'
                  for i in range(len(hulls)))
fi = f"{I[0,0]:.6e} {I[1,1]:.6e} {I[2,2]:.6e} {I[0,1]:.6e} {I[0,2]:.6e} {I[1,2]:.6e}"
xml = f"""<mujoco model="duck">
  <!-- Rubber duck for pick_place_duck. Frame: x beak, y left, z up, origin = COM; {LENGTH*100:.1f} cm long.
       Collision: {len(hulls)} convex hulls (head, beak, left wing, right wing, body). Generated by scripts/assets/make_duck.py. -->
  <compiler angle="radian" />
  <asset>
    <texture name="duck_texture" type="2d" file="duck/texture_map.png" />
    <material name="duck_material" texture="duck_texture" />
    <mesh name="duck_visual_mesh" file="duck/textured.obj" inertia="shell" />
{col}
  </asset>
  <worldbody>
    <body name="duck" pos="0 0 0">
      <freejoint name="duck_joint" />
      <inertial pos="0 0 0" mass="{MASS}" fullinertia="{fi}" />
      <geom name="duck_visual_geom" type="mesh" mesh="duck_visual_mesh" material="duck_material" contype="0" conaffinity="0" group="0" />
{geoms}
    </body>
  </worldbody>
</mujoco>
"""
open(os.path.join(OUT, "duck.xml"), "w").write(xml)
allv = np.vstack([h.vertices for h in hulls.values()]) - com

# head numbers for the controller (visual surface, duck frame, COM origin)
body_vis = trimesh.Trimesh(to_duck(body_shell.vertices) - com, body_shell.faces, process=False)
all_vis = trimesh.Trimesh(to_duck(w.vertices) - com, w.faces, process=False)


def head_slice(z):
    """Segments (n, 2, 3) of the head's horizontal cross-section at height z (x > -20 mm drops the tail)."""
    s = trimesh.intersections.mesh_plane(body_vis, [0, 0, 1], [0, 0, z])
    return s[(s[:, :, 0] > -0.02).all(1)]


def cheeks_at(s, x0):
    """(right, left) cheek y where the cross-section s crosses x = x0."""
    a, b = s[:, 0], s[:, 1]
    m = (a[:, 0] - x0) * (b[:, 0] - x0) <= 0
    dx = b[m, 0] - a[m, 0]
    t = np.divide(x0 - a[m, 0], dx, out=np.zeros_like(dx), where=np.abs(dx) > 1e-12)
    y = a[m, 1] + t * (b[m, 1] - a[m, 1])
    return y.min(), y.max()


s_aim = head_slice(HEAD_LOCAL[2])
width_aim = float(np.ptp(s_aim[:, :, 1]))
x_back, x_face = float(s_aim[:, :, 0].min()), float(s_aim[:, :, 0].max())
width_min = min(float(np.ptp(head_slice(z)[:, :, 1])) for z in np.arange(0.0138, 0.02901, 0.0002))
xs = np.arange(-0.010, 0.0301, 0.0005)


def flat_span(zs):
    """x span around HEAD_LOCAL x where both cheeks stay >= 19 mm from the midplane at every height in zs."""
    half = np.array([[min(-yr, yl) for yr, yl in (cheeks_at(head_slice(z), x) for x in xs)] for z in zs]).min(0)
    i0 = i1 = int(np.argmin(np.abs(xs - HEAD_LOCAL[0])))
    if half[i0] < 0.019:
        return float("nan"), float("nan")
    while i0 > 0 and half[i0 - 1] >= 0.019:
        i0 -= 1
    while i1 < len(xs) - 1 and half[i1 + 1] >= 0.019:
        i1 += 1
    return xs[i0], xs[i1]


flat_aim, flat_band = flat_span([HEAD_LOCAL[2]]), flat_span(np.arange(0.016, 0.02901, 0.001))
rmi = trimesh.ray.ray_triangle.RayMeshIntersector(all_vis)


def body_top(ya):
    """Highest body / wing surface under the cheek-pinch fingertips: rays from above at |y| = ya, x in [-16, 36] mm."""
    o = np.array([[x, sgn * ya, 0.1] for x in np.arange(-0.016, 0.0365, 0.002) for sgn in (1.0, -1.0)])
    loc, _, _ = rmi.intersects_location(o, np.tile([0.0, 0.0, -1.0], (len(o), 1)), multiple_hits=True)
    return float(loc[:, 2].max()) if len(loc) else float("nan")


bk = hulls["beak"].bounds - com
hd = hulls["head"].bounds - com
mm = lambda a: ", ".join(f"{1e3 * float(t):+.1f}" for t in np.ravel(a))
ext = 1e3 * (to_duck(w.vertices).max(0) - to_duck(w.vertices).min(0))
report = [
    f"# duck <- {SRC}/duck.obj (Genesis {getattr(gs, '__version__', '?')}), built by scripts/assets/make_duck.py",
    f"length {LENGTH:.3f} m beak to tail: scale S = {LENGTH:.3f} / {w.extents[2]:.4f} = {S:.4e} m per file unit",
    "duck frame: x beak, y the duck's left, z up; v_duck = S*(Zf, Xf, Yf) - c (file is Y-up, beak +Z), MJCF body unrotated",
    "centre of mass (the origin, collision union) c = (" + ", ".join(f"{1e3 * t:+.3f}" for t in com) + ") mm in the scaled file axes",
    f"visual extents {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm (length x width x height)",
    f"collision bounds (mm): x [{mm(allv[:, 0].min())}, {mm(allv[:, 0].max())}] y [{mm(allv[:, 1].min())}, {mm(allv[:, 1].max())}]"
    f" z [{mm(allv[:, 2].min())}, {mm(allv[:, 2].max())}]",
    f"bottom {-1e3 * allv[:, 2].min():.2f} mm below the COM; xy radius {1e3 * np.linalg.norm(allv[:, :2], axis=1).max():.1f} mm (tail tip)",
    f"collision: {len(hulls)} convex hulls from the mesh's shells (no CoACD), <= {MAX_F} faces each (fast_simplification agg 2, re-hulled);"
    f" head = body shell file Y >= {NECK_YF:g} + both eyes, body = body shell file Y <= {NECK_YF + OVERLAP_YF:g}",
]
report += [f"  collision_{i} {k}: {len(h.vertices)} verts, {len(h.faces)} faces, simplification vs full hull: inward {loss[k]['inward_mm']} mm,"
           f" outward {loss[k]['outward_mm']} mm" for i, (k, h) in enumerate(hulls.items())]
report += [
    f"solid union (0.5 mm grid) {vol * 1e6:.1f} cm3; mass {MASS:.3f} kg (uniform density {MASS / vol:.0f} kg/m3); friction {FRICTION} 0.005 0.0001",
    f"inertia at the COM (kg m2): Ixx {I[0, 0]:.4e} Iyy {I[1, 1]:.4e} Izz {I[2, 2]:.4e} Ixy {I[0, 1]:.2e} Ixz {I[0, 2]:.4e} Iyz {I[1, 2]:.2e}",
    "head, for the controller (visual surface, duck frame, COM origin):",
    f"  head centre HEAD_LOCAL ({mm(HEAD_LOCAL)}) mm: centre of the widest lateral section,"
    f" {1e3 * (HEAD_LOCAL[2] - allv[:, 2].min()):.1f} mm above the table when upright",
    f"  cheek to cheek {1e3 * width_aim:.1f} mm at z {mm(HEAD_LOCAL[2])}; at least {1e3 * width_min:.1f} mm for z +13.8 .. +29.0 mm",
    f"  flat cheek: half-width >= 19 mm for x {mm(flat_aim[0])} .. {mm(flat_aim[1])} mm at z {mm(HEAD_LOCAL[2])},"
    f" for x {mm(flat_band[0])} .. {mm(flat_band[1])} mm at every z +16 .. +29",
    f"  front to back at z {mm(HEAD_LOCAL[2])}: x {mm(x_back)} (back of the head) .. {mm(x_face)} mm (face)",
    f"  beak: x {mm(bk[0, 0])} .. {mm(bk[1, 0])}, z {mm(bk[0, 2])} .. {mm(bk[1, 2])}, |y| <= {1e3 * np.abs(bk[:, 1]).max():.1f} mm",
    f"  head hull z {mm(hd[0, 2])} .. {mm(hd[1, 2])} mm (neck cut at file Y {NECK_YF:g} = z {mm(NECK_YF * S - com[2])}; top of the head)",
    "  body top under the cheeks: " + ", ".join(f"z {mm(body_top(ya))} at |y| {1e3 * ya:.1f}" for ya in (0.0205, 0.024, 0.028, 0.032)) + " mm",
    f"  pinch band z {mm(PINCH_BAND_Z[0])} .. {mm(PINCH_BAND_Z[1])} mm, aim {mm(HEAD_LOCAL[2])}; above about +29 mm the head curves inward",
]
open(os.path.join(OUT, "IMPORT_REPORT.txt"), "w").write("\n".join(report) + "\n")
sha = {f: hashlib.sha256(open(f"{SRC}/{f}", "rb").read()).hexdigest() for f in ("duck.obj", "material.mtl", "material_0.png")}
attribution = f"""Source: Genesis (github.com/Genesis-Embodied-AI/Genesis), bundled asset genesis/assets/meshes/duck/ (the textured rubber duck:
duck.obj, material.mtl, material_0.png), in the Genesis repo since commit 73f9d2d (2024-12-18); copied from the installed
genesis-world {getattr(gs, '__version__', '?')} package. SHA-256: duck.obj {sha['duck.obj']}
  material.mtl {sha['material.mtl']}
  material_0.png {sha['material_0.png']}
Licence: covered only by the Genesis repository's Apache-2.0 licence; where the model originally came from (author, upstream
licence) is not documented in Genesis.
Generated by scripts/assets/make_duck.py; visual OBJ vertices re-expressed in the duck frame (UVs, faces and texture untouched),
MTL map_Kd renamed to texture_map.png, collision hulls and inertia computed from the mesh.
"""
open(os.path.join(OUT, "ATTRIBUTION.txt"), "w").write(attribution)
print("\n".join(report))
