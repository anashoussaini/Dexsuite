"""Repo-local interactive builder for DexSuite environments.

The builder assembles `dexsuite.make` configurations without becoming part of
the installed `dexsuite` runtime package. The runner keeps actions torch-native
and loads optional teleoperation devices only when selected.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
