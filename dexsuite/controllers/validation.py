"""Validate controller configuration values consistently.

The helpers coerce accepted numeric values to Python scalars and raise explicit
errors with the controller name included. Numeric controller gains and limits
are finite values in controller-native units.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def require_bool(controller_name: str, name: str, value: bool | None) -> bool:
    """Return a required boolean controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value.

    Returns:
        Validated boolean value.

    Raises:
        TypeError: When `value` is not a boolean.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{controller_name}: {name} must be a bool, got {value!r}.")
    return value


def require_positive_float(
    controller_name: str,
    name: str,
    value: float | None,
) -> float:
    """Return a required finite positive float controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value.

    Returns:
        Validated positive finite value.

    Raises:
        ValueError: When `value` is not finite or is less than or equal to zero.
    """
    if not _is_finite_number(value) or float(value) <= 0.0:
        raise ValueError(f"{controller_name}: {name} must be > 0, got {value!r}.")
    return float(value)


def require_optional_positive_float(
    controller_name: str,
    name: str,
    value: float | None,
) -> float | None:
    """Return `None` or a finite positive float controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value, or `None`.

    Returns:
        Validated positive finite value, or `None` when no value was supplied.

    Raises:
        ValueError: When `value` is not `None` and is not finite or positive.
    """
    if value is None:
        return None
    return require_positive_float(controller_name, name, value)


def require_nonnegative_float(
    controller_name: str,
    name: str,
    value: float | None,
) -> float:
    """Return a required finite non-negative float controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value.

    Returns:
        Validated finite value greater than or equal to zero.

    Raises:
        ValueError: When `value` is not finite or is less than zero.
    """
    if not _is_finite_number(value) or float(value) < 0.0:
        raise ValueError(f"{controller_name}: {name} must be >= 0, got {value!r}.")
    return float(value)


def require_positive_int(
    controller_name: str,
    name: str,
    value: int | None,
) -> int:
    """Return a required positive integer controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value.

    Returns:
        Validated positive integer value.

    Raises:
        ValueError: When `value` is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{controller_name}: {name} must be a positive int, got {value!r}.",
        )
    return int(value)


def require_optional_positive_int(
    controller_name: str,
    name: str,
    value: int | None,
) -> int | None:
    """Return `None` or a positive integer controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value, or `None`.

    Returns:
        Validated positive integer value, or `None` when no value was supplied.

    Raises:
        ValueError: When `value` is not `None` and is not a positive integer.
    """
    if value is None:
        return None
    return require_positive_int(controller_name, name, value)


def require_limit_gain(controller_name: str, value: float) -> float:
    """Return a configuration-limit gain in the interval `(0, 1]`.

    Args:
        controller_name: Name used in validation error messages.
        value: Candidate gain value.

    Returns:
        Validated gain.

    Raises:
        ValueError: When `value` is not finite or outside `(0, 1]`.
    """
    if not _is_finite_number(value) or not (0.0 < float(value) <= 1.0):
        raise ValueError(
            f"{controller_name}: configuration_limit_gain must be in (0, 1], "
            f"got {value!r}.",
        )
    return float(value)


def require_choice(controller_name: str, name: str, value: T, choices: set[T]) -> T:
    """Return a value if it is one of the allowed choices.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate option value.
        choices: Allowed values for the option.

    Returns:
        Original value when it belongs to `choices`.

    Raises:
        ValueError: When `value` is not in `choices`.
    """
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in sorted(choices))
        raise ValueError(f"{controller_name}: {name} must be one of {allowed}.")
    return value


def require_positive_vector(
    controller_name: str,
    name: str,
    value: Sequence[float] | None,
    *,
    length: int,
) -> tuple[float, ...]:
    """Return a finite positive vector controller option.

    Args:
        controller_name: Name used in validation error messages.
        name: Option name being validated.
        value: Candidate sequence of numeric entries.
        length: Required number of entries.

    Returns:
        Tuple containing `length` positive finite values.

    Raises:
        ValueError: When `value` is missing, has the wrong length, or contains non-positive or non-finite entries.
    """
    if value is None:
        raise ValueError(f"{controller_name}: {name} is required.")
    try:
        vector = tuple(float(entry) for entry in value)
    except TypeError as exc:
        raise ValueError(
            f"{controller_name}: {name} must be a length-{length} sequence.",
        ) from exc

    if len(vector) != length:
        raise ValueError(
            f"{controller_name}: {name} must be a length-{length} sequence.",
        )
    if not all(math.isfinite(entry) and entry > 0.0 for entry in vector):
        raise ValueError(
            f"{controller_name}: {name} entries must be finite and > 0.",
        )
    return vector


def normalize_pose_backend(controller_name: str, backend: str) -> str:
    """Normalize a pose-controller backend name.

    Args:
        controller_name: Name used in validation error messages.
        backend: Backend name or supported alias.

    Returns:
        Canonical backend key.

    Raises:
        ValueError: When `backend` is not a supported pose backend.
    """
    key = str(backend).strip().lower()
    aliases = {
        "mink": "mink",
        "genesis": "genesis_internal",
        "genesis_ik": "genesis_internal",
        "genesis_internal": "genesis_internal",
        "internal": "genesis_internal",
    }
    if key not in aliases:
        raise ValueError(
            f"{controller_name}: backend must be one of 'mink' or 'genesis_internal'.",
        )
    return aliases[key]


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
