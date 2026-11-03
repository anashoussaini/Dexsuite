"""Non-blocking key-to-callback helper.

This helper polls stdin for a single keystroke and triggers a user-provided
callback. It is intended for interactive scripts and teleoperation loops.
"""

from __future__ import annotations

import contextlib
import sys

if sys.platform == "win32":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import select
    import termios
    import tty


class ExternalResetHelper:
    """Call registered callbacks when corresponding keys are pressed."""

    def __init__(self, keys: dict[str, object]):
        self.triggers = dict(keys)
        self._enabled = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self._fd: int | None = None
        self._old_settings = None

        if not self._enabled:
            return

        if sys.platform != "win32":
            self._fd = sys.stdin.fileno()
            try:
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except Exception:
                self._enabled = False

    def key_pressed(self):
        """Poll stdin once and run a callback if a registered key is pressed.

        Returns:
            The pressed key, or None if no key was pressed.
        """
        if not self._enabled:
            return None

        if sys.platform == "win32":  # pragma: no cover
            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
        else:
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if not dr:
                return None
            key = sys.stdin.read(1)

        if key not in self.triggers:
            return None
        func = self.triggers[key]
        if sys.platform != "win32":  # pragma: no cover
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        if callable(func):
            func()
        return key

    def close(self) -> None:
        """Restore the terminal state (POSIX only)."""
        if sys.platform != "win32" and self._fd is not None and self._old_settings:
            with contextlib.suppress(Exception):
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self._enabled = False

    def __del__(self):
        """Best-effort cleanup."""
        with contextlib.suppress(Exception):
            self.close()
