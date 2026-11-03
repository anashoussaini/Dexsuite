"""CLI helpers for automated data-collection scripts."""

from __future__ import annotations

from collections.abc import Sequence

try:
    from scripts.automated_data_collection.planner import Planner
    from scripts.automated_data_collection.utils.runner import run_collection
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from planner import Planner  # type: ignore[no-redef]
    from utils.runner import run_collection  # type: ignore[no-redef]


def run_planner_script(
    planner_cls: type[Planner],
    argv: Sequence[str] | None = None,
) -> int:
    """Run one planner script."""
    return run_collection(planner_cls, argv)
