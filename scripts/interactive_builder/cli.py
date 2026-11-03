"""CLI for the DexSuite interactive builder."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from scripts.interactive_builder.spec import BuilderSpec
else:
    from spec import BuilderSpec


def _default_ui() -> str:
    try:
        __import__("curses")
    except ImportError:
        return "simple"
    else:
        return "tui"


def _add_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ui",
        choices=("tui", "simple"),
        default=_default_ui(),
        help="User interface mode (default: tui if available).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("dexsuite_builder_spec.json"),
        help="Where to write the saved builder spec (JSON).",
    )
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Only build + save; do not launch the environment runner.",
    )
    p.add_argument(
        "--input",
        dest="input_device",
        choices=("keyboard", "spacemouse", "vive_controller", "vive_tracker", "none"),
        default=None,
        help=(
            "Override the teleoperation input device when auto-running after build "
            "(default: use the builder selection)."
        ),
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a builder spec JSON file created by the builder.",
    )
    p.add_argument(
        "--input",
        dest="input_device",
        choices=("keyboard", "spacemouse", "vive_controller", "vive_tracker", "none"),
        default=None,
        help="Override the teleoperation input device from the saved spec.",
    )


def _cmd_build(args: argparse.Namespace) -> int:
    if args.ui == "tui":
        if __package__:
            from scripts.interactive_builder.tui import run_tui
        else:
            from tui import run_tui

        spec = run_tui()
    else:
        if __package__:
            from scripts.interactive_builder.simple import run_simple_builder
        else:
            from simple import run_simple_builder

        spec = run_simple_builder()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec.to_json(args.output)
    print(f"[dexsuite-builder] wrote spec -> {args.output}")

    if args.no_run:
        return 0

    if __package__:
        from scripts.interactive_builder.runner import run_from_spec
    else:
        from runner import run_from_spec

    # CLI flag wins only when explicitly provided; otherwise keep the TUI/simple
    # builder selection saved in the spec.
    input_device = None
    if args.input_device is not None:
        spec.input_device = str(args.input_device)
        spec.to_json(args.output)
        input_device = str(args.input_device)
    return int(run_from_spec(spec, input_device=input_device))


def _cmd_run(args: argparse.Namespace) -> int:
    spec = BuilderSpec.from_json(args.config)
    if __package__:
        from scripts.interactive_builder.runner import run_from_spec
    else:
        from runner import run_from_spec

    return int(run_from_spec(spec, input_device=args.input_device))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive-builder command line interface."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    # If no subcommand is provided, treat it as "build" so build-only flags
    # (like --ui) are actually parsed.
    if not argv_list or argv_list[0] not in {"build", "run"}:
        argv_list = ["build", *argv_list]

    parser = argparse.ArgumentParser(prog="dexsuite-builder")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_build = sub.add_parser("build", help="Launch the interactive builder.")
    _add_build_args(p_build)
    p_build.set_defaults(_fn=_cmd_build)

    p_run = sub.add_parser("run", help="Run from an existing builder spec.")
    _add_run_args(p_run)
    p_run.set_defaults(_fn=_cmd_run)

    args = parser.parse_args(argv_list)
    return int(args._fn(args))
