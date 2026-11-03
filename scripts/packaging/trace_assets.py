#!/usr/bin/env python
"""Find the asset files the dexsuite PyPI package ships.

Every shipped task and every robot is built once (one process each). The model files Genesis is asked to load and the
files Python opens under dexsuite/models are recorded, then followed through their references (MJCF includes, meshes
and textures, URDF meshes, OBJ materials and texture maps, glTF uris).

    PYOPENGL_PLATFORM=egl python scripts/packaging/trace_assets.py [--jobs 4]

writes scripts/packaging/assets.txt: every file a shipped task or robot loads, and the floor textures. Files under
dexsuite/models that no shipped task or robot loads (the rest of the object library) stay in the
repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "dexsuite" / "models"
OUT = Path(__file__).resolve().parent

TRACK1 = ["reach", "lift", "push", "pick_place", "stack"]
BIMANUAL_BASE = ["bimanual_reach", "bimanual_lift", "bimanual_push", "bimanual_pick_place", "bimanual_stack"]
TRACK1_GRIPPERS = ["robotiq", "allegro", "umi"]
# tasks whose models stay out of the wheel: the window alone would take it over PyPI's 100 MB file limit; the task runs
# from a clone of the repository
FROM_SOURCE = {"close_window_spray"}
# tasks built for one robot are traced with it (the others with Franka + Allegro)
FLOATING_SHARPA = dict(manipulator="floating", gripper="sharpa")
TASK_ROBOT = {t: FLOATING_SHARPA for t in ("spray_graffiti", "put_bread_toaster", "pick_place_duck", "open_safe", "stow_in_safe",
                                          "pliers_in_jewelry_box", "close_window_spray")}


def specs() -> list[dict]:
    """(task, make kwargs) for every shipped task and robot."""
    s = [dict(task=t, kw=dict(manipulator="franka", gripper=g)) for t in TRACK1 for g in TRACK1_GRIPPERS]
    s += [dict(task=t, kw=dict(manipulator=["franka", "franka"], gripper=["allegro_left", "allegro_right"]))
          for t in BIMANUAL_BASE]
    tasks = sorted({m.group(1) for f in (REPO / "dexsuite/environments").rglob("*.py")
                    for m in re.finditer(r'@register_env\("([a-z0-9_]+)"\)', f.read_text())})
    for t in sorted(set(tasks) - FROM_SOURCE):
        if t.startswith("bimanual"):
            s.append(dict(task=t, kw=dict(manipulator=["franka", "franka"], gripper=["allegro_left", "allegro_right"])))
        else:
            s.append(dict(task=t, kw=dict(TASK_ROBOT.get(t, dict(manipulator="franka", gripper="allegro")))))
    arms = ["franka", "franka_fr3", "gen3", "kuka", "link6", "lite6", "sawyer", "ur5", "ur10", "z1"]
    s += [dict(task="reach", kw=dict(manipulator=a, gripper="robotiq")) for a in arms]
    grippers = sorted({m.group(1) for f in (MODELS / "grippers").rglob("*.py")
                       for m in re.finditer(r'@register_gripper\("([a-z0-9_]+)"\)', f.read_text())})
    for arm in ("franka", "floating"):                            # every end-effector each one mounts
        adapters = set(_yaml_keys(MODELS / f"manipulators/{arm}/{arm}_adapters.yaml"))
        s += [dict(task="reach", kw=dict(manipulator=arm, gripper=g)) for g in grippers if g in adapters]
    integrated = ["franka_with_gripper", "aloha_tactile", "gen3_lite", "so_100", "trossen_vx300s", "trossen_wx250s"]
    s += [dict(task="reach", kw=dict(manipulator=r)) for r in integrated]
    return s


def _yaml_keys(path: Path) -> list[str]:
    return [m.group(1) for m in re.finditer(r"^([A-Za-z0-9_]+)\s*:", path.read_text(), re.M)]


# --------------------------------------------------------------------------------------------- worker (one build)
def worker(spec_json: str) -> None:
    import genesis as gs

    seen: set[str] = set()

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            try:
                rp = str(Path(p).resolve())
            except (OSError, ValueError):
                return
            if rp.startswith(str(MODELS)):
                seen.add(rp)

    sys.addaudithook(hook)
    for name in ("MJCF", "URDF", "Mesh"):
        cls = getattr(gs.morphs, name)
        orig = cls.__init__

        def wrapped(self, *a, __orig=orig, **k):
            if k.get("file"):
                seen.add(str(Path(str(k["file"])).resolve()))
            return __orig(self, *a, **k)

        cls.__init__ = wrapped
    import dexsuite as ds

    spec = json.loads(spec_json)
    err = ""
    try:
        env = ds.make(spec["task"], cameras=None, **spec["kw"])
        env.reset(seed=0)
    except Exception as e:  # noqa: BLE001  reported, the build of the lists stops on it
        err = f"{type(e).__name__}: {e}"[:300]
    print("TRACE " + json.dumps({"err": err, "files": sorted(seen)}), flush=True)


# --------------------------------------------------------------------------------------------- reference expansion
def _mjcf_refs(path: Path, meshdir: Path | None = None, texdir: Path | None = None) -> list[Path]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    base, out = path.parent, []
    for c in root.iter("compiler"):
        a = c.get("assetdir")
        if c.get("meshdir") or a:
            meshdir = base / (c.get("meshdir") or a)
        if c.get("texturedir") or a:
            texdir = base / (c.get("texturedir") or a)
    mdir, tdir = meshdir or base, texdir or base
    for el in root.iter():
        if el.tag == "include" and el.get("file"):
            p = (base / el.get("file")).resolve()
            out += [p, *_mjcf_refs(p, meshdir, texdir)]
        elif el.tag in ("mesh", "skin", "hfield") and el.get("file"):
            out.append((mdir / el.get("file")).resolve())
        elif el.tag == "texture":
            out += [(tdir / el.get(k)).resolve() for k in ("file", "fileright", "fileleft", "fileup", "filedown",
                                                            "filefront", "fileback") if el.get(k)]
    return out


def _urdf_refs(path: Path) -> list[Path]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    out = []
    for m in root.iter("mesh"):
        f = re.sub(r"^package://[^/]+/", "", m.get("filename") or "")
        if f:
            cands = [path.parent / f, path.parent.parent / f]
            out.append(next((c for c in cands if c.exists()), cands[0]).resolve())
    return out


def _obj_refs(path: Path) -> list[Path]:
    out = []
    try:
        lines = path.read_text(errors="ignore").splitlines()[:5000]
    except OSError:
        return out
    for line in lines:
        if line.startswith("mtllib"):
            mtl = (path.parent / line.split(None, 1)[1].strip()).resolve()
            out.append(mtl)
            if mtl.exists():
                for ml in mtl.read_text(errors="ignore").splitlines():
                    parts = ml.strip().split()
                    if parts and parts[0].lower() in ("map_kd", "map_ks", "map_ka", "map_bump", "bump", "map_d", "norm",
                                                      "map_ns", "disp", "map_pr", "map_pm"):
                        out.append((mtl.parent / parts[-1]).resolve())
    return out


def _gltf_refs(path: Path) -> list[Path]:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [(path.parent / x["uri"]).resolve() for k in ("buffers", "images") for x in d.get(k, [])
            if "uri" in x and not x["uri"].startswith("data:")]


def expand(files: set[Path]) -> set[Path]:
    todo, used = list(files), set()
    while todo:
        p = todo.pop()
        if p in used or not str(p).startswith(str(MODELS)):
            continue
        used.add(p)
        s = p.suffix.lower()
        refs = {".xml": _mjcf_refs, ".urdf": _urdf_refs, ".obj": _obj_refs, ".gltf": _gltf_refs}.get(s, lambda _: [])(p)
        todo += [r for r in refs if r not in used]
    return {p for p in used if p.is_file()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--worker")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return
    all_specs = specs()

    def run(spec):
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(spec)], capture_output=True, text=True,
                           cwd="/tmp", timeout=1800)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("TRACE ")), None)
        d = json.loads(line[6:]) if line else {"err": (r.stderr or r.stdout)[-300:], "files": []}
        print(("FAILED " if d["err"] else "ok     ") + spec["task"], spec["kw"], d["err"][:120], flush=True)
        return spec, d

    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(run, all_specs))
    failed = [(s, d["err"]) for s, d in results if d["err"]]
    if failed:
        raise SystemExit(f"{len(failed)} builds failed, lists not written")
    used = expand({Path(f) for _, d in results for f in d["files"]})
    textures = {p.resolve() for p in (MODELS / "textures").rglob("*") if p.is_file()}   # selectable floors
    config = (".py", ".pyc", ".yaml", ".yml")                     # code and configs ship as package data
    assets = {p for p in used | textures if p.suffix not in config}
    (OUT / "assets.txt").write_text("\n".join(sorted(str(p.relative_to(MODELS.resolve())) for p in assets)) + "\n")
    print(f"{len(assets)} files {sum(p.stat().st_size for p in assets) / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
