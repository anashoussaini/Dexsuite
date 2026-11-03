from __future__ import annotations

from scripts.automated_data_collection import lift_automated


def test_lift_default_segment_action_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["lift_automated.py"])

    args = lift_automated.parse_args()

    assert args.move_to_cube_steps == 100
    assert args.close_steps == 5
    assert args.settle_steps == 2
    assert args.lift_steps == 50
    assert args.hold_steps == 10
    assert (
        args.move_to_cube_steps
        + args.close_steps
        + args.settle_steps
        + args.lift_steps
        + args.hold_steps
    ) == 167
