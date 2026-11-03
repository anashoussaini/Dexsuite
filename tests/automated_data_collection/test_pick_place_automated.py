from __future__ import annotations

from scripts.automated_data_collection import pick_place_automated


def test_pick_place_default_segment_action_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pick_place_automated.py"])

    args = pick_place_automated.parse_args()

    planned_steps = (
        args.move_to_cube_steps
        + args.close_steps
        + args.settle_steps
        + args.transport_steps
        + args.release_steps
        + args.retreat_steps
    )

    assert args.max_steps == 200
    assert args.move_to_cube_steps == 100
    assert args.close_steps == 5
    assert args.settle_steps == 2
    assert args.transport_steps == 40
    assert args.transport_release_start is None
    assert args.release_steps == 5
    assert args.retreat_steps == 8
    assert planned_steps == 160
    assert planned_steps < args.max_steps
