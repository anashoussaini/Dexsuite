from __future__ import annotations

from argparse import Namespace

import torch

from scripts.automated_data_collection.planner import tcp_pos
from scripts.automated_data_collection.reach_automated import ReachPlanner
from scripts.automated_data_collection.utils.runner import _tune_args, make_parser


def test_reach_uses_tcp_control_point():
    planner = ReachPlanner(Namespace(), profile=object())

    assert planner.control_pos is tcp_pos


def test_reach_tcp_target_conversion_is_identity():
    planner = ReachPlanner(Namespace(), profile=object())
    target = torch.tensor([0.1, 0.2, 0.3])

    assert planner.tcp_target_for_control_target({}, target) is target


def test_reach_default_segment_budget_is_short():
    args = make_parser(ReachPlanner).parse_args([])

    _tune_args(args)

    assert args.move_steps == 60
    assert args.max_steps == 100
    assert args.motion_tolerance == 0.020
