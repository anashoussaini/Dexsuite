from __future__ import annotations

import importlib

from dexsuite.core.registry import ENV_REG

ARTICULATED_TASKS = {
    "put_bread_toaster",
    "open_safe",
    "stow_in_safe",
    "spray_graffiti",
    "close_window_spray",
    "pliers_in_jewelry_box",
}


def test_articulated_tasks_are_registered() -> None:
    importlib.import_module("dexsuite.environments")
    missing = ARTICULATED_TASKS - set(ENV_REG)
    assert not missing, f"unregistered tasks: {sorted(missing)}"
