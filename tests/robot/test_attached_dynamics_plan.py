"""CPU contracts for the attachment repair's supported layouts and API spans."""
from types import SimpleNamespace

import numpy as np
import pytest

from dexsuite.utils.attached_dynamics import _attachment_plan


def _scene_specs(specs):
    """Create model handles with separate immutable-by-contract local ranges."""
    entities, links = [], []
    next_dof = next_geom = 0
    for index, (name, n_links, n_dofs, n_geoms, parent_entity) in enumerate(specs):
        entity = SimpleNamespace(
            idx=index, dof_start=next_dof, dof_end=next_dof + n_dofs,
            q_start=next_dof, q_end=next_dof + n_dofs,
            link_start=len(links), link_end=len(links) + n_links,
            geom_start=next_geom, geom_end=next_geom + n_geoms,
            gravity_compensation=1.0, is_local_collision_mask=True, links=[],
        )
        for local_index in range(n_links):
            parent = len(links) - 1
            if local_index == 0:
                parent = -1 if parent_entity is None else entities[parent_entity].link_end - 1
            link = SimpleNamespace(
                name=name if local_index == 0 else f"{name}_{local_index}",
                parent_idx=parent, entity=entity,
            )
            entity.links.append(link)
            links.append(link)
        entities.append(entity)
        next_dof += n_dofs
        next_geom += n_geoms
    return SimpleNamespace(entities=entities, links=links)


def _floating_hand():
    return _scene_specs([
        ("table", 1, 0, 1, None),
        ("gantry_x", 6, 6, 0, None),
        ("right_hand_C_MC", 23, 22, 33, 1),
        ("task_object", 2, 7, 2, None),
    ])


def test_attached_hand_forms_one_solver_tree_without_changing_robot_handles():
    solver = _floating_hand()
    before = [(e.idx, e.q_start, e.q_end, e.dof_start, e.dof_end,
               e.link_start, e.link_end, e.geom_start, e.geom_end) for e in solver.entities]
    plan = _attachment_plan(solver)
    assert plan["groups"] == [[1, 2]]
    np.testing.assert_array_equal(plan["owners"], [0, 1, 1, 3])
    np.testing.assert_array_equal(plan["spans"]["dof"], [[0, 0, 28, 28], [0, 28, 28, 35]])
    np.testing.assert_array_equal(plan["spans"]["link"], [[0, 1, 30, 30], [1, 30, 30, 32]])
    np.testing.assert_array_equal(plan["spans"]["geom"], [[0, 1, 34, 34], [1, 34, 34, 36]])
    after = [(e.idx, e.q_start, e.q_end, e.dof_start, e.dof_end,
              e.link_start, e.link_end, e.geom_start, e.geom_end) for e in solver.entities]
    assert after == before


def test_unattached_and_unrelated_attached_robots_keep_native_path():
    solver = _floating_hand()
    solver.entities[2].links[0].parent_idx = -1
    assert _attachment_plan(solver) is None
    other = _scene_specs([
        ("other_arm", 6, 6, 4, None),
        ("other_hand", 4, 3, 4, 0),
    ])
    assert _attachment_plan(other) is None


def test_nonadjacent_attachment_is_rejected_before_swallowing_task_object():
    solver = _scene_specs([
        ("gantry_x", 6, 6, 0, None),
        ("task_object", 1, 1, 1, None),
        ("right_hand_C_MC", 23, 22, 33, 0),
    ])
    with pytest.raises(RuntimeError, match="contiguous solver ranges"):
        _attachment_plan(solver)


@pytest.mark.parametrize("span", ["dof", "link", "geom"])
def test_noncontiguous_storage_is_rejected(span):
    solver = _floating_hand()
    hand = solver.entities[2]
    setattr(hand, span + "_start", getattr(hand, span + "_start") + 1)
    with pytest.raises(RuntimeError, match=f"noncontiguous {span} spans"):
        _attachment_plan(solver)


@pytest.mark.parametrize("property_name", ["gravity_compensation", "is_local_collision_mask"])
def test_incompatible_group_properties_are_not_silently_overridden(property_name):
    solver = _floating_hand()
    setattr(solver.entities[2], property_name, 0)
    with pytest.raises(RuntimeError, match=f"mixed {property_name}"):
        _attachment_plan(solver)


def test_extra_attached_body_on_target_tree_is_rejected():
    solver = _scene_specs([
        ("gantry_x", 6, 6, 0, None),
        ("right_hand_C_MC", 23, 22, 33, 0),
        ("extra_tool", 1, 1, 1, 1),
    ])
    with pytest.raises(RuntimeError, match="one attached hand"):
        _attachment_plan(solver)
