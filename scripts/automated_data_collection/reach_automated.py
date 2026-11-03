"""Run automated data collection for the reach task."""

from __future__ import annotations

from collections.abc import Sequence

try:
    from scripts.automated_data_collection.planner import (
        ObjectTarget,
        Planner,
        Primitive,
        tcp_pos,
    )
    from scripts.automated_data_collection.utils.cli import run_planner_script
except ModuleNotFoundError:  # pragma: no cover - direct file execution fallback.
    from planner import (  # type: ignore[no-redef]
        ObjectTarget,
        Planner,
        Primitive,
        tcp_pos,
    )
    from utils.cli import run_planner_script  # type: ignore[no-redef]


class ReachPlanner(Planner):
    """Planner implementation for reach."""

    task = "reach"

    @property
    def control_pos(self):
        """Use the TCP itself for reach success alignment."""
        return tcp_pos

    def tcp_target_for_control_target(self, _obs: dict, control_target):
        """Reach targets already refer to the TCP."""
        return control_target

    def build_primitives(self, _obs: dict) -> Sequence[Primitive]:
        """Return the reach primitive."""
        return [
            Primitive(
                "reach:target",
                ObjectTarget("target_pos"),
                self.args.move_steps,
                0.0,
                0.0,
                tolerance=self.args.motion_tolerance,
            ),
        ]


def main() -> int:
    """Run the reach collector."""
    return run_planner_script(ReachPlanner)


if __name__ == "__main__":
    raise SystemExit(main())
