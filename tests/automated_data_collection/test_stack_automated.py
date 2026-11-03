from __future__ import annotations

from scripts.automated_data_collection import stack_automated


def test_stack_default_segment_action_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["stack_automated.py"])

    args = stack_automated.parse_args()

    per_cube_steps = (
        args.approach_transition_steps
        + args.move_to_cube_steps
        + args.close_steps
        + args.settle_steps
        + args.transport_steps
        + args.release_steps
        + args.retreat_steps
    )

    assert args.max_steps == 520
    assert args.move_to_cube_steps == 100
    assert args.approach_transition_steps == 20
    assert args.close_steps == 5
    assert args.settle_steps == 2
    assert args.transport_steps == 60
    assert args.release_steps == 5
    assert args.retreat_steps == 20
    assert per_cube_steps == 212
    assert 2 * per_cube_steps < args.max_steps
