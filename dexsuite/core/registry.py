"""Provide central registries and decorators for runtime object discovery.

This module exposes several dictionary-based registries that map string keys to
classes, allowing for flexible lookup of environments, manipulators, grippers,
and controllers. It also provides the decorator functions used to populate
these registries.

Attributes:
    ENV_REG: Maps a task name (e.g., 'reach') to an environment class.
    MANIP_REG: Maps a manipulator name (e.g., 'franka') to an ArmModel class.
    GRIP_REG: Maps a gripper name (e.g., 'allegro') to a GripperModel class.
    CONTROLLER_REG: Maps a controller name (e.g., 'joint_position') to a
        controller class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)

ENV_REG: dict[str, type] = {}
MANIP_REG: dict[str, type] = {}
GRIP_REG: dict[str, type] = {}
CONTROLLER_REG: dict[str, type] = {}


# --------------------------------------------------------------------- #
# Decorators
# --------------------------------------------------------------------- #
def register_controller(key: str) -> Callable[[T], T]:
    """Decorate a controller class to add it to the registry.

    Args:
        key: The case-insensitive name used for lookup (e.g., 'joint_position' or 'cartesian_impedance').

    Returns:
        A decorator that adds the class to the CONTROLLER_REG.
    """
    key = key.lower()

    def _decorator(cls: T) -> T:
        CONTROLLER_REG[key] = cls
        cls._registry_name = key
        return cls

    return _decorator


def register_env(key: str) -> Callable[[T], T]:
    """Decorate an environment class to make it available for creation.

    Args:
        key: The case-insensitive task identifier (e.g., 'reach').

    Returns:
        A decorator that adds the class to the ENV_REG.
    """
    key = key.lower()

    def _decorator(cls: T) -> T:
        ENV_REG[key] = cls
        cls._registry_name = key
        return cls

    return _decorator


def register_manipulator(key: str) -> Callable[[T], T]:
    """Decorate an ArmModel subclass to add it to the registry.

    Args:
        key: The case-insensitive manipulator identifier (e.g., 'franka' or 'ur5').

    Returns:
        A decorator that adds the class to the MANIP_REG.
    """
    key = key.lower()

    def _decorator(cls: T) -> T:
        MANIP_REG[key] = cls
        cls._registry_name = key
        return cls

    return _decorator


def register_gripper(key: str) -> Callable[[T], T]:
    """Decorate a GripperModel subclass to add it to the registry.

    Args:
        key: The case-insensitive gripper identifier (e.g., 'allegro' or 'robotiq85').

    Returns:
        A decorator that adds the class to the GRIP_REG.
    """
    key = key.lower()

    def _decorator(cls: T) -> T:
        GRIP_REG[key] = cls
        cls._registry_name = key
        return cls

    return _decorator
