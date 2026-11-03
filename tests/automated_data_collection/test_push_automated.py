from __future__ import annotations

from scripts.automated_data_collection import push_automated


def test_push_default_segment_action_budget(monkeypatch):
    monkeypatch.setattr("sys.argv", ["push_automated.py"])

    args = push_automated.parse_args()

    planned_steps = (
        args.approach_steps
        + args.contact_steps
        + args.push_steps
        + args.hold_steps
    )

    assert args.max_steps == 240
    assert args.approach_steps == 80
    assert args.contact_steps == 20
    assert args.push_steps == 70
    assert args.hold_steps == 10
    assert planned_steps == 180
    assert planned_steps < args.max_steps
