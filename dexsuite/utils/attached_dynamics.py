"""Correct Genesis 0.4.3 dynamics ownership for floating right Sharpa hands.

Genesis attach() joins the link tree but leaves separate entity ranges. Its
per-entity CRB reduction consequently omits the hand payload from upstream
wrist links, and its block mass matrix omits wrist/finger coupling. Coalescing
only the solver's ranges restores one physical tree while keeping the two
public robot handles and their local indices.
"""
from __future__ import annotations

import math
from types import MethodType

import numpy as np


def _attachment_plan(solver):
    """Plan supported contiguous groups without modifying model/API objects."""
    entities = solver.entities
    owners = {id(entity): i for i, entity in enumerate(entities)}
    tree_owners = list(range(len(entities)))
    for child, entity in enumerate(entities):
        base = entity.links[0]
        if base.parent_idx < 0:
            continue
        parent = owners[id(solver.links[base.parent_idx].entity)]
        if parent != child:
            if parent >= child:
                raise RuntimeError("Attached dynamics requires parent-first entities")
            tree_owners[child] = tree_owners[parent]
    groups = []
    for root in sorted(set(tree_owners)):
        group = [i for i, owner in enumerate(tree_owners) if owner == root]
        names = [entities[i].links[0].name for i in group]
        # Keep unrelated robot/model arrangements on their existing path.
        if not names or names[0] != "gantry_x" or "right_hand_C_MC" not in names:
            continue
        if names != ["gantry_x", "right_hand_C_MC"]:
            raise RuntimeError("Floating Sharpa dynamics supports one attached hand per wrist")
        if group != list(range(group[0], group[-1] + 1)):
            raise RuntimeError("Floating Sharpa entities must have contiguous solver ranges")
        for prop in ("gravity_compensation", "is_local_collision_mask"):
            if any(getattr(entities[i], prop) != getattr(entities[root], prop) for i in group):
                raise RuntimeError(f"Floating Sharpa attachment has mixed {prop}")
        groups.append(group)
    if not groups:
        return None

    physics_owners = np.arange(len(entities), dtype=np.int32)
    spans = {}
    for name in ("dof", "link", "geom"):
        starts = np.array([getattr(e, name + "_start") for e in entities], dtype=np.int32)
        ends = np.array([getattr(e, name + "_end") for e in entities], dtype=np.int32)
        for group in groups:
            if any(ends[a] != starts[b] for a, b in zip(group, group[1:])):
                raise RuntimeError(f"Floating Sharpa attachment has noncontiguous {name} spans")
            end = int(ends[group[-1]])
            ends[group[0]] = end
            for child in group[1:]:
                starts[child] = ends[child] = end
                physics_owners[child] = group[0]
        spans[name] = (starts, ends)
    public = [{key: getattr(e, key) for key in (
        "idx", "dof_start", "dof_end", "q_start", "q_end",
        "link_start", "link_end", "geom_start", "geom_end",
    )} for e in entities]
    return {"owners": physics_owners, "groups": groups, "spans": spans, "public_ranges": public}


def _collision_filter_inputs(solver):
    # These are the Python model inputs used by native collider pair filtering.
    # Runtime physics ownership must not alter geometry/entity identities.
    return tuple((
        id(g), id(g.entity), g.link.idx, g.link.root_idx, g.link.is_fixed,
        g.contype, g.conaffinity, g.entity.is_local_collision_mask,
        g.friction, g.type,
    ) for g in solver.geoms)


def prepare_attached_dynamics(scene) -> bool:
    """Install build hooks on this solver only, immediately before scene.build.

    The compatibility repair is restricted to Genesis 0.4.3 and contiguous
    floating/right-Sharpa pairs. Other robot arrangements and engine versions
    are left unchanged. Unsupported modes on the affected pair fail explicitly.
    """
    import genesis as gs
    from genesis.utils.misc import qd_to_numpy

    solver = scene.sim.rigid_solver
    if gs.__version__ != "0.4.3" or getattr(solver, "_dexsuite_attached_dynamics_prepared", False):
        return False
    # An explicitly installed diagnostic prototype already supplies this repair.
    if getattr(type(solver), "_attachment_prototype_installed", False):
        return False
    plan = _attachment_plan(solver)
    if plan is None:
        return False
    original_config = solver._build_static_config
    original_fields = solver._init_entity_fields

    def configure(self):
        original_config()
        config = self._static_rigid_sim_config
        if config.requires_grad or config.use_hibernation or config.enable_heterogeneous:
            raise RuntimeError("Floating Sharpa dynamics repair does not support gradients, hibernation or heterogeneous models")
        max_dofs = int(max(plan["spans"]["dof"][1] - plan["spans"]["dof"][0]))
        config.max_n_dofs_per_entity = max_dofs
        config.max_n_links_per_entity = int(max(plan["spans"]["link"][1] - plan["spans"]["link"][0]))
        config.max_n_geoms_per_entity = int(max(plan["spans"]["geom"][1] - plan["spans"]["geom"][0]))
        if gs.backend != gs.cpu:
            import quadrants as qd
            shared_kb = 32.0 if gs.backend == gs.metal else 48.0
            max_warps = int(math.sqrt(shared_kb * 1024 / (4 if gs.qd_float == qd.f32 else 8))) // 32
            config.enable_tiled_cholesky_mass_matrix = 8 <= max_dofs <= max_warps * 32 and self.n_envs <= 16384
            config.tiled_n_dofs_per_entity = min(max(math.ceil(max_dofs / 32), 1), max_warps) * 32

    def initialize_fields(self):
        original_fields()
        collision_inputs = _collision_filter_inputs(self)
        for name, (starts, ends) in plan["spans"].items():
            getattr(self.entities_info, name + "_start").from_numpy(starts)
            getattr(self.entities_info, name + "_end").from_numpy(ends)
            plural = {"dof": "dofs", "link": "links", "geom": "geoms"}[name]
            getattr(self.entities_info, "n_" + plural).from_numpy(ends - starts)
        for info in (self.links_info, self.dofs_info):
            entity_ids = np.array(qd_to_numpy(info.entity_idx), copy=True)
            info.entity_idx.from_numpy(plan["owners"][entity_ids].astype(entity_ids.dtype))
        # Imported MuJoCo weights describe the separate pre-attachment models.
        # Recompute every affected link and DOF using native undamped inertia.
        for name, info in (("link", self.links_info), ("dof", self.dofs_info)):
            weights = np.array(qd_to_numpy(info.invweight), copy=True)
            starts, ends = plan["spans"][name]
            for group in plan["groups"]:
                weights[starts[group[0]]:ends[group[0]]] = -1
            info.invweight.from_numpy(weights)
        # Native per-entity FK/IK dispatch also needs the physical tree owner.
        for i, entity in enumerate(self.entities):
            if plan["owners"][i] != i:
                entity._idx_in_solver = int(plan["owners"][i])
                for link in entity.links:
                    link._entity_idx_in_solver = int(plan["owners"][i])
        public_unchanged = all(
            all(getattr(entity, key) == value for key, value in before.items())
            for entity, before in zip(self.entities, plan["public_ranges"])
        )
        collision_unchanged = _collision_filter_inputs(self) == collision_inputs
        if not public_unchanged or not collision_unchanged:
            raise RuntimeError("Attached dynamics unexpectedly changed public ranges or collision filtering")
        self._dexsuite_attached_dynamics_audit = {
            "genesis_version": gs.__version__,
            "groups": plan["groups"],
            "owners": plan["owners"].tolist(),
            "runtime_spans": {name: {"start": a.tolist(), "end": b.tolist()} for name, (a, b) in plan["spans"].items()},
            "public_ranges_preserved": public_unchanged,
            "collision_filter_inputs_preserved": collision_unchanged,
            "max_n_dofs_per_entity": self._static_rigid_sim_config.max_n_dofs_per_entity,
            "tiled_n_dofs_per_entity": self._static_rigid_sim_config.tiled_n_dofs_per_entity,
        }

    solver._build_static_config = MethodType(configure, solver)
    solver._init_entity_fields = MethodType(initialize_fields, solver)
    solver._dexsuite_attached_dynamics_prepared = True
    return True
