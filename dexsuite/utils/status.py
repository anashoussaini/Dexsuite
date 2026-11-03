"""Display status information with Rich or ASCII fallbacks.

Rich output is used only for interactive terminals unless environment variables disable it. Spinner output writes to stderr and falls back to plain lines when animation is unavailable.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

import torch

from .globals import get_device

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAVE_RICH = True
except ImportError:
    _HAVE_RICH = False
    Console = None


# --- Capabilities & ANSI ---

RESET = "\033[0m"
BLUE = "\033[38;5;39m"
GREEN = "\033[38;5;82m"
RED = "\033[38;5;203m"

HDR_BG = "#183153"
HDR_BORD = "#2563EB"
LBL_FG = "#60A5FA"
HDR_FG = "white"


def _is_tty(file) -> bool:
    """Return whether a file stream is connected to a TTY."""
    try:
        return bool(file.isatty())
    except AttributeError:
        return False


def _use_rich_ui() -> bool:
    """Return whether Rich UI output should be used."""
    flag = os.environ.get("DEXSUITE_STATUS", "").strip().lower()
    if flag in {"0", "false", "off"}:
        return False
    if flag in {"1", "true", "on"}:
        return _HAVE_RICH
    return _HAVE_RICH and _is_tty(sys.stderr)


def _use_spinner() -> bool:
    """Return whether an animated spinner should be shown."""
    flag = os.environ.get("DEXSUITE_SPINNER", "").strip().lower()
    if flag in {"0", "false", "off"}:
        return False
    return _is_tty(sys.stderr)


# --- Version ---


def _package_version(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def resolve_version() -> str:
    """Resolve the installed DexSuite package version.

    Returns:
        The package version string or "dev" if not found.
    """
    for name in ("dexsuite", "DexSuite"):
        installed_version = _package_version(name)
        if installed_version is not None:
            return installed_version
    mod = sys.modules.get("dexsuite")
    v = getattr(mod, "__version__", None) if mod else None
    return v if isinstance(v, str) and v else "dev"


# --- Pretty Labels ---

_LABEL_ALIASES = {
    "Device": "Compute Device",
    "n_envs": "Number of Environments",
    "control_hz": "Control Rate",
    "substeps": "Physics Substeps",
    "Mode": "Robot Mode",
    "Robot": "Robot",
    "Controllers": "Controllers",
    "Left Robot": "Left Robot",
    "Left Controllers": "Left Controllers",
    "Right Robot": "Right Robot",
    "Right Controllers": "Right Controllers",
}
_LABEL_ICONS = {
    "Compute Device": "🖥",
    "Number of Environments": "∑",
    "Control Rate": "⏱",
    "Physics Substeps": "⚙",
    "Robot Mode": "🤖",
}


def _pretty_label(k: str) -> str:
    """Formats a configuration key into a pretty label, potentially with an icon.

    Args:
        k: The raw configuration key (e.g., "n_envs").

    Returns:
        The formatted label string (e.g., "∑ Number of Environments").
    """
    k2 = _LABEL_ALIASES.get(k, k.replace("_", " ").title())
    icon = _LABEL_ICONS.get(k2, "")
    return f"{icon} {k2}" if icon else k2


def _device_string() -> str:
    """Return a descriptive string for the current compute device."""
    dev = get_device()
    device_name = str(dev)
    try:
        if getattr(dev, "type", None) == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
    except (AssertionError, RuntimeError):
        return device_name
    return device_name


def _partition_intro_details(
    details: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[Text]]:
    """Split intro details into compact-table rows and long standalone rows."""
    short_details: list[tuple[str, str]] = []
    long_renderables: list[Text] = []
    long_keys = {
        "Robot",
        "Controllers",
        "Left Robot",
        "Left Controllers",
        "Right Robot",
        "Right Controllers",
    }

    for key, value in details:
        pretty_key = _pretty_label(key)
        if key in long_keys:
            long_renderables.append(
                Text.assemble(
                    (f"{pretty_key}: ", f"bold {LBL_FG}"),
                    (str(value), "bold"),
                ),
            )
        else:
            short_details.append((key, value))

    return short_details, long_renderables


# --- Intro Panel ---


def _table_2col_pairs(kv: Sequence[tuple[str, str]]) -> Table:
    """Builds a Rich Table grid with 4 columns for a 2x2 key-value layout.

    Args:
        kv: A sequence of (key, value) string pairs for short details.

    Returns:
        A Rich Table object.
    """
    table = Table.grid(padding=(0, 4))
    for _ in range(4):
        table.add_column(no_wrap=True)

    items = [(_pretty_label(k), str(v)) for (k, v) in kv]
    if len(items) % 2 == 1:
        items.append(("", ""))

    for i in range(0, len(items), 2):
        (k1, v1), (k2, v2) = items[i], items[i + 1]
        table.add_row(
            Text(k1, style=f"bold {LBL_FG}"),
            Text(v1, style="bold"),
            Text(k2, style=f"bold {LBL_FG}"),
            Text(v2, style="bold"),
        )
    return table


def show_intro(
    *,
    title: str = "DexSuite",
    subtitle: str | None = None,
    details: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Display a centered introductory panel.

    Separates short details (2x2 grid) from long details (full rows).

    Args:
        title: The main title for the panel header.
        subtitle: An optional subtitle string.
        details: An optional sequence of (key, value) pairs for configuration details.
    """
    if not _use_rich_ui():
        print(f"=== {title} v{resolve_version()} ===")
        if subtitle:
            print(subtitle)
        if details:
            for k, v in details:
                print(f"  {_pretty_label(k)}: {v}")
        return

    console = Console(file=sys.stderr, force_terminal=True, soft_wrap=False)

    header = Text.assemble(
        (" ", ""),
        (f"{title} v{resolve_version()} ", f"bold {HDR_FG} on {HDR_BG}"),
    )

    content_renderables: list[Any] = []

    if subtitle:
        content_renderables.append(
            Align.center(Text(subtitle, style="bold", justify="center")),
        )
        if details:
            content_renderables.append(Text())

    if details:
        short_details, long_items_renderables = _partition_intro_details(details)

        if short_details:
            content_renderables.append(_table_2col_pairs(short_details))
            if long_items_renderables:
                content_renderables.append(Text())

        content_renderables.extend(long_items_renderables)

    content = Group(*content_renderables) if content_renderables else ""

    panel = Panel(content, title=header, border_style=HDR_BORD, padding=(1, 4))
    console.print(Align.center(panel))
    console.print()


def _safe_name(x: Any, default: str = "none") -> str:
    """Safely gets a name string from an object."""
    if x is None:
        return default
    option_name = getattr(x, "name", None)
    if isinstance(option_name, str) and option_name:
        return option_name

    controller = getattr(x, "ctrl", None)
    if controller is not None:
        controller_name = getattr(controller, "name", None)
        if isinstance(controller_name, str) and controller_name:
            return controller_name

    cls_name = getattr(type(x), "__name__", None)
    if cls_name and cls_name != "NoneType":
        return cls_name

    return str(x) if str(x) else default


def _robot_summary(robot_options: Any) -> dict:
    """Return a summary dictionary for robot configuration options.

    Args:
        robot_options: The RobotOptions object.

    Returns:
        A dictionary summarizing the robot mode, components, and controllers.
    """
    typ = getattr(robot_options, "type_of_robot", "single")
    mode_name = typ.capitalize()
    summary = {"Mode": mode_name}

    if typ == "bimanual":
        left = getattr(robot_options, "left", None)
        right = getattr(robot_options, "right", None)

        if left:
            left_manip = getattr(left, "manipulator", "?")
            left_grip = getattr(left, "gripper", "builtin")
            summary["Left Robot"] = f"{left_manip}+{left_grip}"
            left_manip_ctrl = _safe_name(getattr(left, "manipulator_controller", None))
            left_grip_ctrl = _safe_name(getattr(left, "gripper_controller", None))
            summary["Left Controllers"] = f"{left_manip_ctrl}/{left_grip_ctrl}"
        else:
            summary["Left Robot"] = "?"
            summary["Left Controllers"] = "?"

        if right:
            right_manip = getattr(right, "manipulator", "?")
            right_grip = getattr(right, "gripper", "builtin")
            summary["Right Robot"] = f"{right_manip}+{right_grip}"
            right_manip_ctrl = _safe_name(
                getattr(right, "manipulator_controller", None),
            )
            right_grip_ctrl = _safe_name(getattr(right, "gripper_controller", None))
            summary["Right Controllers"] = f"{right_manip_ctrl}/{right_grip_ctrl}"
        else:
            summary["Right Robot"] = "?"
            summary["Right Controllers"] = "?"
    else:
        single = getattr(robot_options, "single", None)
        if single:
            manip = getattr(single, "manipulator", "?")
            grip = getattr(single, "gripper", "builtin")
            summary["Robot"] = f"{manip}+{grip}"
            manip_ctrl = _safe_name(getattr(single, "manipulator_controller", None))
            grip_ctrl = _safe_name(getattr(single, "gripper_controller", None))
            summary["Controllers"] = f"{manip_ctrl}/{grip_ctrl}"
        else:
            summary["Robot"] = "?"
            summary["Controllers"] = "?"
    return summary


def show_intro_env(
    *,
    robot_options: Any,
    sim: Any,
    render_mode: str | None = None,
    sim_dt: float | None = None,
    substeps: int | None = None,
) -> None:
    """Display the environment setup introductory panel.

    Args:
        robot_options: The RobotOptions object used for configuration.
        sim: The simulation object (expected to have n_envs, control_hz).
        render_mode: The rendering mode string (ignored).
        sim_dt: Simulation timestep (ignored).
        substeps: Number of physics substeps.
    """
    del render_mode, sim_dt
    n_envs = getattr(sim, "n_envs", "?")
    control_hz = getattr(sim, "control_hz", "?")

    robot_info = _robot_summary(robot_options)

    details = [
        ("Device", _device_string()),
        ("n_envs", str(n_envs)),
        ("control_hz", f"{control_hz} Hz"),
        (
            "substeps",
            str(substeps)
            if substeps is not None
            else str(getattr(sim, "substeps", "1")),
        ),
        ("Mode", robot_info.get("Mode", "?")),
    ]

    if robot_info.get("Mode") == "Bimanual":
        details.extend(
            [
                ("Left Robot", robot_info.get("Left Robot", "?")),
                ("Left Controllers", robot_info.get("Left Controllers", "?")),
                ("Right Robot", robot_info.get("Right Robot", "?")),
                ("Right Controllers", robot_info.get("Right Controllers", "?")),
            ],
        )
    else:
        details.extend(
            [
                ("Robot", robot_info.get("Robot", "?")),
                ("Controllers", robot_info.get("Controllers", "?")),
            ],
        )

    show_intro(
        title="DexSuite",
        subtitle="Welcome - initializing the DexSuite runtime",
        details=details,
    )


# --- Spinner ---


@contextmanager
def building(label: str = "Building"):
    """Context manager to display a spinner during long operations.

    Args:
        label: Text to display next to the spinner.
    """
    start = time.perf_counter()
    spin = _LineSpinner(label)
    try:
        spin.start()
        yield
    except BaseException:
        spin.stop(success=False, elapsed=time.perf_counter() - start)
        raise
    else:
        spin.stop(success=True, elapsed=time.perf_counter() - start)


class _LineSpinner:
    """Manage spinner-thread output for long operations."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, label: str, interval: float = 0.08) -> None:
        self.label = label
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None
        self._animate = _use_spinner()
        self._file = sys.stderr

    def start(self) -> None:
        """Starts the spinner animation in a background thread if enabled."""
        if not self._animate:
            self._println(f"{BLUE}… {self.label}{RESET}")
            return
        if self._thr is not None:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self, *, success: bool, elapsed: float) -> None:
        """Stops the spinner and prints the final status line.

        Args:
            success: Whether the operation succeeded (True) or failed (False).
            elapsed: The duration of the operation in seconds.
        """
        if self._animate and self._thr is not None:
            self._stop.set()
            self._thr.join(timeout=0.5)
        icon = "✔" if success else "✖"
        color = GREEN if success else RED
        self._write_line(f"{icon} {self.label} ({elapsed:.2f}s)", color, final=True)

    def _run(self) -> None:
        """Run the spinner animation loop in a background thread."""
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            i += 1
            self._write_line(f"{frame} {self.label}", BLUE, final=False)
            time.sleep(self.interval)

    def _write_line(self, text: str, color: str, final: bool) -> None:
        """Writes/overwrites the spinner line to stderr."""
        msg = f"{color}{text}{RESET}"
        self._file.write(
            "\r\x1b[2K" + msg,
        )  # Carriage return, erase line, write message
        self._file.flush()
        if final:
            self._file.write("\n")
            self._file.flush()

    def _println(self, text: str) -> None:
        """Prints a simple line to stderr (used when animation is disabled)."""
        self._file.write(text + "\n")
        self._file.flush()


# --- Action Terms Table ---


def _pretty_action_name(name: tuple[str, ...]) -> str:
    """Formats an action segment's hierarchical name for display."""
    parts = list(name)
    if parts and parts[-1] == "manipulator":
        parts[-1] = "arm_action"
    elif parts and parts[-1] == "gripper":
        parts[-1] = "gripper_action"
    return ".".join(parts)


def render_action_terms_ascii(layout: Any) -> str:
    """Render an ASCII table summarizing the action layout.

    Args:
        layout: The ActionLayout object.

    Returns:
        A string containing the formatted ASCII table.
    """
    title = f"Action Terms (shape: {getattr(layout, 'total_dim', '?')})"
    rows: list[str] = []

    max_start_idx_len = 1
    segments = getattr(layout, "segments", [])
    if segments:
        max_start_idx = max(getattr(seg, "start", 0) for seg in segments)
        max_start_idx_len = len(str(max_start_idx))

    idx_width = max(5, max_start_idx_len)
    name_width = 20
    dim_width = 3

    header_line = (
        f"| {'Index':<{idx_width}} | {'Name':<{name_width}} | {'Dim':<{dim_width}} |"
    )
    sep_line = (
        "+"
        + "-" * (idx_width + 2)
        + "+"
        + "-" * (name_width + 2)
        + "+"
        + "-" * (dim_width + 2)
        + "+"
    )
    total_width = len(sep_line) - 2

    rows.append("+" + "-" * total_width + "+")
    pad = (total_width - len(title)) // 2
    pad_rem = total_width - len(title) - pad
    rows.append("|" + " " * pad + title + " " * pad_rem + "|")
    rows.append(sep_line)
    rows.append(header_line)
    rows.append(sep_line)

    for seg in segments:
        start_idx = getattr(seg, "start", "?")
        nm = _pretty_action_name(getattr(seg, "name", ()))
        dim = getattr(seg, "width", lambda: "?")()
        # Ensure dim is handled correctly if it's not a number
        try:
            dim_int = int(dim)
            dim_str = f"{dim_int:<{dim_width}}"
        except (ValueError, TypeError):
            dim_str = f"{'?':<{dim_width}}"

        rows.append(f"| {start_idx:<{idx_width}} | {nm:<{name_width}} | {dim_str} |")

    rows.append(sep_line)
    return "\n".join(rows)


def show_action_terms(layout: Any, *, title: str = "Action Terms") -> None:
    """Display a table summarizing the action layout.

    Args:
        layout: The ActionLayout object.
        title: The title for the table panel.
    """
    if not _use_rich_ui():
        print(render_action_terms_ascii(layout))
        return

    console = Console(file=sys.stderr, force_terminal=True)
    tbl = Table(show_header=True, header_style=f"bold {LBL_FG}")
    tbl.add_column("Index", justify="right", no_wrap=True)
    tbl.add_column("Name", justify="left")
    tbl.add_column("Dim", justify="right", no_wrap=True)

    for seg in getattr(layout, "segments", []):
        start_idx = getattr(seg, "start", "?")
        nm = _pretty_action_name(getattr(seg, "name", ()))
        dim = getattr(seg, "width", lambda: "?")()
        # Ensure dim is a string for Rich
        try:
            dim_str = str(int(dim))
        except (ValueError, TypeError):
            dim_str = "?"

        tbl.add_row(str(start_idx), nm, dim_str)

    panel_title = Text(
        f"{title} (shape: {getattr(layout, 'total_dim', '?')})",
        style=f"bold {HDR_FG} on {HDR_BG}",
    )
    block = Panel(tbl, title=panel_title, border_style=HDR_BORD, padding=(1, 2))
    console.print(Align.center(block))
