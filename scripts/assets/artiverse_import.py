"""Turn an Artiverse ``urdf_w_collider`` model into a Dexsuite object asset.

Artiverse URDFs (metres, Z-up, one root link, ``link_<pid>_<label>`` / ``joint_<child>_to_<parent>``)
ship placeholder physics: every link has mass 1.0 with a unit-mass geometric inertia tensor placed at the
link (= joint) origin rather than the part's centre of mass, joints
have no ``<dynamics>`` and generic effort limits, and the visual OBJs double as collision meshes
(Genesis turns each into one convex hull, so cavities and slots disappear). The shipped
``collision_meshes/`` are in the raw source frame, not the URDF frame, and are ignored here.

``import_model`` copies the visual OBJ/MTL (+ textures), rewrites inertials (the requested mass
spread uniformly over the convex hull of the link's visuals: consistent CoM and tensor; ``LinkSpec.com``
moves the CoM, e.g. a liquid-filled bottle, transporting the hull tensor there), joint types / limits / efforts / dynamics, and rebuilds every link's
collision set from a :class:`LinkSpec`:

* ``"hull"``       - the visual meshes as-is (one Genesis hull per OBJ);
* ``"merged_hull"``- one convex hull of all visuals, written to ``collision/<link>.obj``;
* ``"coacd"``      - offline CoACD decomposition of the merged visuals, ``collision/<link>/<k>.obj``;
* ``"primitives"`` - hand-authored ``<box>`` / ``<cylinder>`` elements in the link frame;
* ``"none"``       - no collision geometry.

Genesis scales ``<box size>``, prismatic limits, inertials (mass x s^3, inertia x s^5) and mesh
vertices with ``morph.scale``; ``<dynamics>`` values are not scaled. ``LinkSpec.mesh_scale`` stretches one
link's meshes per axis about the visual mesh origin, before the <visual><origin> transform: written as ``<mesh scale>`` on its visuals (and hull-mode
collisions); computed hulls and the inertial use the scaled vertices.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

# the Artiverse dataset (huggingface.co/datasets/3dlg-hcvc/artiverse), downloaded; set DEXSUITE_ARTIVERSE_ROOT to its data/ folder
ARTIVERSE_ROOT = Path(os.environ.get("DEXSUITE_ARTIVERSE_ROOT", "artiverse/data"))
OBJECTS_ROOT = Path(__file__).resolve().parents[2] / "dexsuite" / "models" / "objects"


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def from_bounds(lo, hi) -> Box:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        return Box(tuple((0.5 * (lo + hi)).tolist()), tuple((hi - lo).tolist()))


@dataclass(frozen=True)
class Cylinder:
    center: tuple[float, float, float]
    radius: float
    length: float
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)  # cylinder axis = local z after rpy


@dataclass
class LinkSpec:
    mass: float | None = None
    collision: str = "hull"
    primitives: list[Box | Cylinder] = field(default_factory=list)
    coacd_threshold: float = 0.05
    coacd_max_hulls: int = 8
    visual_primitives: list[Box | Cylinder] = field(default_factory=list)  # added parts (also collide)
    visual_rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    com: tuple[float | None, float | None, float | None] | None = None  # link-frame CoM override; None entries keep the hull's
    mesh_scale: tuple[float, float, float] | None = None  # per-axis factor on every <mesh> of the link (about the visual mesh origin, before the <visual><origin> transform)


@dataclass
class JointSpec:
    type: str | None = None
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    damping: float | None = None
    friction: float | None = None


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3)
    return np.array([float(v) for v in text.split()], dtype=float)


def _mesh_scale(mesh_el: ET.Element) -> np.ndarray:
    return _parse_xyz(mesh_el.get("scale")) if mesh_el.get("scale") else np.ones(3)


def _scale_mesh(mesh_el: ET.Element, scale) -> None:
    mesh_el.set("scale", _fmt(_mesh_scale(mesh_el) * np.asarray(scale, dtype=float)))


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def find_model_urdf(category: str, source: str, model_id: str) -> Path:
    urdfs = sorted((ARTIVERSE_ROOT / category / source / model_id / "urdf_w_collider").glob("*.urdf"))
    if len(urdfs) != 1:
        raise FileNotFoundError(f"expected one URDF for {category}/{source}/{model_id}, found {urdfs}")
    return urdfs[0]


def link_visual_mesh(urdf_path: Path, link: ET.Element) -> trimesh.Trimesh:
    """All visual meshes of ``link`` merged (``<mesh scale>`` applied), expressed in the link frame."""
    parts = []
    for vis in link.findall("visual"):
        origin = vis.find("origin")
        xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
        rot = _rpy_to_matrix(_parse_xyz(origin.get("rpy") if origin is not None else None))
        mesh_el = vis.find("geometry/mesh")
        if mesh_el is None:
            continue
        m = trimesh.load(urdf_path.parent / mesh_el.get("filename"), force="mesh", process=False)
        m = trimesh.Trimesh(vertices=(m.vertices * _mesh_scale(mesh_el)) @ rot.T + xyz, faces=m.faces, process=False)
        parts.append(m)
    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)


def _copy_visual(urdf_path: Path, mesh_el: ET.Element, dst_objs: Path) -> None:
    src = (urdf_path.parent / mesh_el.get("filename")).resolve()
    dst_objs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_objs / src.name)
    mtl = src.with_suffix(".mtl")
    if mtl.exists():
        shutil.copy2(mtl, dst_objs / mtl.name)
        for line in mtl.read_text(errors="ignore").splitlines():
            if line.strip().lower().startswith("map_"):
                tex = (mtl.parent / line.split(None, 1)[1].strip()).resolve()
                if tex.exists():
                    rel = tex.relative_to(src.parent)
                    (dst_objs / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tex, dst_objs / rel)
    mesh_el.set("filename", f"objs/{src.name}")


def _recolor_mtls(dst_objs: Path, by_material: dict[str, tuple], by_file_prefix: dict[str, tuple]) -> None:
    """Rewrite ``Kd`` of the copied MTLs: per material name, or per OBJ stem prefix (wins)."""
    for mtl in sorted(dst_objs.glob("*.mtl")):
        stem = mtl.name.split("__")[0]
        file_color = next((c for k, c in by_file_prefix.items() if stem.startswith(k)), None)
        out, current = [], None
        for line in mtl.read_text(errors="ignore").splitlines():
            if line.startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
            elif line.startswith("Kd ") and (file_color or current in by_material):
                c = file_color or by_material[current]
                line = f"Kd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"
            out.append(line)
        mtl.write_text("\n".join(out) + "\n")


def _collision_element(*, origin_xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    col = ET.Element("collision")
    ET.SubElement(col, "origin", xyz=_fmt(origin_xyz), rpy=_fmt(rpy))
    return col


def _mesh_collision(filename: str) -> ET.Element:
    col = _collision_element()
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "mesh", filename=filename)
    return col


def _primitive_collision(prim: Box | Cylinder) -> ET.Element:
    col = _collision_element(origin_xyz=prim.center, rpy=prim.rpy)
    geom = ET.SubElement(col, "geometry")
    if isinstance(prim, Box):
        ET.SubElement(geom, "box", size=_fmt(prim.size))
    else:
        ET.SubElement(geom, "cylinder", radius=f"{prim.radius:.6g}", length=f"{prim.length:.6g}")
    return col


def _rebuild_collisions(urdf_path: Path, link: ET.Element, spec: LinkSpec, dst_dir: Path) -> list[str]:
    name = link.get("name")
    visuals = list(link.findall("visual"))
    notes: list[str] = []
    for col in link.findall("collision"):
        link.remove(col)
    if spec.collision == "none":
        return notes
    if spec.collision == "hull":
        for vis in visuals:
            col = ET.Element("collision")
            origin = vis.find("origin")
            ET.SubElement(col, "origin", xyz=origin.get("xyz", "0 0 0"), rpy=origin.get("rpy", "0 0 0"))
            geom = ET.SubElement(col, "geometry")
            src_mesh = vis.find("geometry/mesh")
            mesh_el = ET.SubElement(geom, "mesh", filename=src_mesh.get("filename"))
            if src_mesh.get("scale"):
                mesh_el.set("scale", src_mesh.get("scale"))
            link.append(col)
        notes.append(f"{name}: {len(visuals)} visual hull(s)")
        return notes
    if spec.collision == "primitives":
        for prim in spec.primitives:
            link.append(_primitive_collision(prim))
        notes.append(f"{name}: {len(spec.primitives)} primitive(s)")
        return notes
    merged = link_visual_mesh(urdf_path, link)
    if spec.collision == "merged_hull":
        hull = merged.convex_hull
        out = dst_dir / "collision" / f"{name}.obj"
        out.parent.mkdir(parents=True, exist_ok=True)
        hull.export(out)
        link.append(_mesh_collision(f"collision/{name}.obj"))
        notes.append(f"{name}: merged hull ({len(hull.faces)} faces)")
        return notes
    if spec.collision == "coacd":
        import coacd

        coacd.set_log_level("error")
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(merged.vertices, dtype=np.float64), np.asarray(merged.faces, dtype=np.int64)),
            threshold=spec.coacd_threshold,
            max_convex_hull=spec.coacd_max_hulls,
            seed=0,
        )
        out_dir = dst_dir / "collision" / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for k, (v, f) in enumerate(parts):
            piece = trimesh.Trimesh(vertices=v, faces=f, process=False).convex_hull
            piece.export(out_dir / f"{name}_{k}.obj")  # stems must be unique: Genesis keys mesh assets by basename
            link.append(_mesh_collision(f"collision/{name}/{name}_{k}.obj"))
        notes.append(f"{name}: {len(parts)} CoACD piece(s), threshold {spec.coacd_threshold}")
        return notes
    raise ValueError(f"unknown collision mode {spec.collision!r} for link {name}")


def _add_visual_primitives(link: ET.Element, spec: LinkSpec) -> None:
    for i, prim in enumerate(spec.visual_primitives):
        vis = ET.Element("visual")
        ET.SubElement(vis, "origin", xyz=_fmt(prim.center), rpy=_fmt(prim.rpy))
        geom = ET.SubElement(vis, "geometry")
        if isinstance(prim, Box):
            ET.SubElement(geom, "box", size=_fmt(prim.size))
        else:
            ET.SubElement(geom, "cylinder", radius=f"{prim.radius:.6g}", length=f"{prim.length:.6g}")
        mat = ET.SubElement(vis, "material", name=f"{link.get('name')}_added_{i}")
        ET.SubElement(mat, "color", rgba=_fmt(spec.visual_rgba))
        link.append(vis)
        link.append(_primitive_collision(prim))


def _apply_inertial(link: ET.Element, mass: float | None, merged: trimesh.Trimesh | None, com=None) -> None:
    """Replace the placeholder inertial with a consistent one: the requested mass distributed uniformly over the
    convex hull of the link's visuals (origin at the hull's centre of mass, tensor about it). With ``com`` the
    origin moves there and the hull tensor is transported by the parallel-axis theorem."""
    inertial = link.find("inertial")
    if inertial is None or mass is None:
        return
    if merged is None or len(merged.vertices) < 4:
        inertial.find("mass").set("value", f"{float(mass):.6g}")
        return
    hull = merged.convex_hull
    if hull.volume <= 1e-12:
        inertial.find("mass").set("value", f"{float(mass):.6g}")
        return
    tensor = hull.moment_inertia * (float(mass) / hull.volume)  # trimesh assumes unit density
    center = np.asarray(hull.center_mass, dtype=float)
    if com is not None:
        center = np.array([hull.center_mass[k] if v is None else float(v) for k, v in enumerate(com)])
        d = center - hull.center_mass
        tensor = tensor + float(mass) * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    origin = inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", _fmt(center))
    origin.set("rpy", "0 0 0")
    inertial.find("mass").set("value", f"{float(mass):.6g}")
    inertia = inertial.find("inertia")
    for key, (i, j) in {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2), "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}.items():
        inertia.set(key, f"{float(tensor[i, j]):.6g}")


def _apply_joint(joint: ET.Element, spec: JointSpec) -> None:
    for attr in list(joint.attrib):
        if attr.startswith("artiverse_"):
            del joint.attrib[attr]
    if spec.type is not None:
        joint.set("type", spec.type)
        if spec.type == "fixed":
            for tag in ("axis", "limit", "dynamics"):
                el = joint.find(tag)
                if el is not None:
                    joint.remove(el)
            return
    limit = joint.find("limit")
    if limit is None and any(v is not None for v in (spec.lower, spec.upper, spec.effort, spec.velocity)):
        limit = ET.SubElement(joint, "limit")
    if limit is not None:
        if spec.lower is not None:
            limit.set("lower", f"{spec.lower:.6g}")
        if spec.upper is not None:
            limit.set("upper", f"{spec.upper:.6g}")
        if spec.effort is not None:
            limit.set("effort", f"{spec.effort:.6g}")
        if spec.velocity is not None:
            limit.set("velocity", f"{spec.velocity:.6g}")
    if spec.damping is not None or spec.friction is not None:
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping", f"{spec.damping or 0.0:.6g}")
        dyn.set("friction", f"{spec.friction or 0.0:.6g}")


def import_model(
    src_urdf: Path,
    dst_dir: Path,
    name: str,
    *,
    links: dict[str, LinkSpec],
    joints: dict[str, JointSpec],
    attribution: str,
    default_link: LinkSpec | None = None,
    mtl_colors: dict[str, tuple] | None = None,
    mtl_colors_by_file: dict[str, tuple] | None = None,
) -> Path:
    """Write ``dst_dir/<name>.urdf`` (+ ``objs/``, ``collision/``, ``ATTRIBUTION.txt``) and return its path."""
    src_urdf = Path(src_urdf)
    if Path(dst_dir).is_symlink():
        raise ValueError(f"refusing to write through a symlink: {dst_dir}")
    dst_dir = Path(dst_dir).resolve()
    if dst_dir.parent.parent != OBJECTS_ROOT.resolve():
        raise ValueError(f"refusing to write outside {OBJECTS_ROOT}/<group>/<name>: {dst_dir}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    tree = ET.parse(src_urdf)
    robot = tree.getroot()
    robot.set("name", name)

    link_names = [l.get("name") for l in robot.findall("link")]
    unknown = set(links) - set(link_names)
    if unknown:
        raise KeyError(f"unknown links {sorted(unknown)}; model has {link_names}")
    joint_names = [j.get("name") for j in robot.findall("joint")]
    unknown = set(joints) - set(joint_names)
    if unknown:
        raise KeyError(f"unknown joints {sorted(unknown)}; model has {joint_names}")

    report: list[str] = [f"# {name} <- {src_urdf}"]
    for link in robot.findall("link"):
        lname = link.get("name")
        spec = links.get(lname, default_link or LinkSpec())
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is not None:
                _copy_visual(src_urdf, mesh_el, dst_dir / "objs")
                if spec.mesh_scale is not None:
                    _scale_mesh(mesh_el, spec.mesh_scale)
        # collisions may reference the source visuals: rebuild before rewriting paths
        merged = link_visual_mesh(src_urdf, link)
        report.extend(_rebuild_collisions(src_urdf, link, spec, dst_dir))
        _add_visual_primitives(link, spec)
        _apply_inertial(link, spec.mass, merged, spec.com)
        if len(merged.vertices):
            lo, hi = merged.bounds
            report.append(
                f"  {lname}: mass {spec.mass}, visual bounds (link frame) lo {np.round(lo, 4).tolist()} hi {np.round(hi, 4).tolist()}"
                + (f", mesh scale {list(spec.mesh_scale)}" if spec.mesh_scale is not None else "")
            )
    if mtl_colors or mtl_colors_by_file:
        _recolor_mtls(dst_dir / "objs", mtl_colors or {}, mtl_colors_by_file or {})
        report.append(f"  recoloured MTLs: materials {sorted((mtl_colors or {}))}, files {sorted((mtl_colors_by_file or {}))}")
    for joint in robot.findall("joint"):
        jname = joint.get("name")
        if jname in joints:
            _apply_joint(joint, joints[jname])
        lim = joint.find("limit")
        dyn = joint.find("dynamics")
        report.append(
            f"  {jname}: {joint.get('type')} {joint.find('parent').get('link')} -> {joint.find('child').get('link')}"
            + (f" limits [{lim.get('lower')}, {lim.get('upper')}] effort {lim.get('effort')}" if lim is not None else "")
            + (f" damping {dyn.get('damping')} friction {dyn.get('friction')}" if dyn is not None else "")
        )
    mesh_files = [m.get("filename") for m in robot.iter("mesh")]
    by_stem: dict[str, set[str]] = {}
    for f in mesh_files:
        by_stem.setdefault(Path(f).stem, set()).add(f)
    clashes = {k: sorted(v) for k, v in by_stem.items() if len(v) > 1}
    if clashes:
        raise ValueError(f"mesh basenames must be unique (Genesis keys assets by stem): {clashes}")
    ET.indent(tree, space="  ")
    out = dst_dir / f"{name}.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    (dst_dir / "ATTRIBUTION.txt").write_text(attribution.strip() + "\n")
    (dst_dir / "IMPORT_REPORT.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"wrote {out}")
    return out


def attribution_text(category: str, source: str, model_id: str, note: str) -> str:
    return f"""Source: Artiverse (3dlg-hcvc/artiverse, Hugging Face), model {category}/{source}/{model_id}
Viewer: https://aspis.cmpt.sfu.ca/artiverse_viewer/#/m/{category}/{source}/{model_id}
Licence: as the upstream {source} asset (see the Artiverse dataset card); research use.
Generated by scripts/assets/{note}; visual OBJ/MTL copied (Kd recoloured where noted), physics and collision rewritten.
"""


def strip_artiverse_ids(text: str) -> str:
    return re.sub(r"[0-9a-f]{32,64}", lambda m: m.group(0)[:8], text)
