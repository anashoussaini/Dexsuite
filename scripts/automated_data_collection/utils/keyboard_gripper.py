"""Asynchronous keyboard gripper override for automated collectors."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KeyboardGripperControl:
    """Maintain a non-blocking keyboard-controlled gripper alpha.

    DexSuite's `Keyboard` device owns an async listener, so polling
    `alpha_for_step()` does not block the scripted arm trajectory. The returned
    alpha replaces only the scripted gripper command; the arm target sequence is
    left unchanged.
    """

    enabled: bool
    control_hz: int
    speed: float
    initial_alpha: float
    layout: str
    invert: bool = False
    alpha: float = field(init=False)
    keyboard: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Clamp the initial manual gripper state."""
        self.alpha = _clamp01(self.initial_alpha)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> KeyboardGripperControl:
        """Build a control state from common CLI arguments."""
        return cls(
            enabled=getattr(args, "gripper_mode", "auto") == "keyboard",
            control_hz=max(int(getattr(args, "control_hz", 20)), 1),
            speed=max(float(getattr(args, "keyboard_gripper_speed", 1.0)), 0.0),
            initial_alpha=float(getattr(args, "keyboard_gripper_initial_alpha", 0.0)),
            layout=str(getattr(args, "keyboard_layout", "qwerty")),
            invert=bool(getattr(args, "keyboard_gripper_invert", False)),
        )

    def start(self) -> None:
        """Start the async keyboard listener when keyboard mode is enabled."""
        if not self.enabled or self.keyboard is not None:
            return
        from dexsuite.devices import Keyboard
        from dexsuite.devices.keyboard import MAPPING_AZERTY, MAPPING_QWERTY

        mapping = MAPPING_QWERTY if self.layout == "qwerty" else MAPPING_AZERTY
        self.keyboard = Keyboard(
            controller="OSCPose",
            normalized=True,
            pos_sens=1.0,
            rot_sens=1.0,
            mapping=mapping,
            gripper=True,
        )
        print(
            "keyboard_gripper=on keys=o/p adjust gripper "
            f"initial_alpha={self.alpha:.3f} speed={self.speed:.3f}/s",
        )

    def alpha_for_step(self, scripted_alpha: float) -> float:
        """Return the gripper alpha to use for the next environment step."""
        if not self.enabled:
            return _clamp01(scripted_alpha)
        if self.keyboard is None:
            self.start()
        command = self.keyboard.get_action() if self.keyboard is not None else {}
        if _first_float(command.get("reset", [0.0])) != 0.0:
            self.alpha = _clamp01(scripted_alpha)
            return self.alpha

        delta = _first_float(command.get("gripper", [0.0]))
        if self.invert:
            delta = -delta
        self.alpha = _clamp01(self.alpha + delta * self.speed / float(self.control_hz))
        return self.alpha

    def close(self) -> None:
        """Stop the keyboard listener if it was created."""
        if self.keyboard is None:
            return
        listener = getattr(self.keyboard, "listener", None)
        if listener is not None:
            listener.stop()
        self.keyboard = None


def add_keyboard_gripper_args(
    parser: argparse.ArgumentParser,
    *,
    include_mode: bool = True,
) -> None:
    """Register shared keyboard gripper CLI flags."""
    if include_mode:
        parser.add_argument(
            "--gripper-mode",
            choices=("auto", "keyboard"),
            default="auto",
            help="auto follows the script; keyboard replaces the gripper slice from o/p input.",
        )
    parser.add_argument(
        "--keyboard-layout",
        choices=("qwerty", "azerty"),
        default="qwerty",
        help="Keyboard layout used by the DexSuite Keyboard device.",
    )
    parser.add_argument(
        "--keyboard-gripper-speed",
        type=float,
        default=1.0,
        help="Manual gripper alpha change per second while holding o/p.",
    )
    parser.add_argument(
        "--keyboard-gripper-initial-alpha",
        type=float,
        default=0.0,
        help="Initial manual gripper alpha, where 0 is open and 1 is closed.",
    )
    parser.add_argument(
        "--keyboard-gripper-invert",
        action="store_true",
        help="Invert the sign of the o/p gripper command if the layout feels reversed.",
    )


def wrap_step_target_for_keyboard(
    fn: Callable,
    gripper_control: KeyboardGripperControl,
) -> Callable:
    """Wrap a direct collector `_step_target` with keyboard gripper control."""
    if not gripper_control.enabled:
        return fn

    def wrapped(
        env,
        obs: dict,
        slices,
        low,
        high,
        logger,
        frame,
        tcp_target,
        *,
        grip_alpha: float,
        trace,
        episode: int,
        step_idx: int,
        phase: str,
    ):
        return fn(
            env,
            obs,
            slices,
            low,
            high,
            logger,
            frame,
            tcp_target,
            grip_alpha=grip_alpha,
            trace=trace,
            episode=episode,
            step_idx=step_idx,
            phase=phase,
            gripper_control=gripper_control,
        )

    return wrapped


def _first_float(value: Any) -> float:
    try:
        return float(value[0])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
