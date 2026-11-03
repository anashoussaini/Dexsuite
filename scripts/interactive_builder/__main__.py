"""Module entry point: python -m scripts.interactive_builder."""

from __future__ import annotations

if __package__:
    from scripts.interactive_builder.cli import main
else:
    from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
