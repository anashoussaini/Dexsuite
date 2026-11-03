"""Non-blocking terminal key helper for triggering resets.

This helper is used by interactive loops to poll a single key without blocking.
It supports both POSIX terminals and Windows consoles.
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


class ResetHelper:
    def __init__(self, key: str):
        self.trigger = str(key)
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
        """Return True when the configured key is pressed."""
        if not self._enabled:
            return False

        if sys.platform == "win32":  # pragma: no cover
            if not msvcrt.kbhit():
                return False
            return msvcrt.getwch() == self.trigger

        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if not dr:
            return False

        key = sys.stdin.read(1)
        if key == self.trigger:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            return True
        return False

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
