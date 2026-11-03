"""Manage process-local runtime state shared across DexSuite.

The global device and environment count are mutable process-level defaults used by controllers, models, and utilities. They are not thread-local, so concurrent simulations should set them deliberately before construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _RuntimeState:
    device: torch.device
    n_envs: int = 1


_STATE = _RuntimeState(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


def set_device(device: str | torch.device) -> None:
    """Globally override the torch / Genesis device.

    Args:
        device: The new device to use (e.g., "cpu", "cuda:0", torch.device("cuda")).
    """
    _STATE.device = torch.device(device)


def get_device() -> torch.device:
    """Return the global device used throughout DexSuite.

    Returns:
        The globally configured torch.device.
    """
    return _STATE.device


def set_n_envs(n: int) -> None:
    """Globally override the number of parallel environments.

    Args:
        n: The new number of environments. Must be a positive integer.
    """
    _STATE.n_envs = int(n)


def get_n_envs() -> int:
    """Return the global number of parallel environments.

    Returns:
        The globally configured integer number of environments.
    """
    return _STATE.n_envs
