"""Create robot controllers from configuration objects.

Controller modules are imported lazily so registration only happens when a
specific controller name is requested. Configuration dictionaries are validated
against constructor keyword parameters before instantiation.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, Protocol

from dexsuite.core.registry import CONTROLLER_REG

__all__ = ["make"]


class _ControllerConfig(Protocol):
    name: str
    config: dict[str, Any]


def _lazy_ensure(key: str) -> None:
    """Import and register a controller module when needed.

    Args:
        key: The name of the controller to load.

    Raises:
        KeyError: If the controller module cannot be found or if the module
            is found but does not register the controller.
    """
    if key not in CONTROLLER_REG:
        try:
            import_module(f"{__name__}.{key}")
        except ModuleNotFoundError as e:
            raise KeyError(
                f"unknown controller '{key}' (no module {__name__}.{key})",
            ) from e
        if key not in CONTROLLER_REG:
            raise KeyError(
                f"unknown controller '{key}' (module imported but not registered)",
            )


def _allowed_from_init(cls: type[object]) -> set[str]:
    """Return controller constructor keyword names accepted from config.

    Args:
        cls: The controller class to inspect.

    Returns:
        Allowed keyword argument names.
    """
    sig = inspect.signature(cls.__init__)
    allowed: set[str] = set()
    for name, p in sig.parameters.items():
        if name in ("self", "entity", "model"):
            continue
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            allowed.add(name)
    return allowed


def make(
    cfg: _ControllerConfig,
    *,
    entity: object,
    model: object,
    **kwargs: object,
) -> Any:
    """Create a controller instance from a configuration object.

    Controller modules are loaded on demand and `cfg.config` is filtered against
    the selected constructor. Extra runtime keyword arguments are rejected so
    callers cannot silently pass unused configuration.

    Args:
        cfg: Controller configuration object with `name` and `config` fields.
        entity: Simulation entity the controller will operate on.
        model: Robot component model associated with the entity.
        kwargs: Reserved for future use. Must be empty.

    Returns:
        Instantiated controller object.

    Raises:
        TypeError: When extra keyword arguments are provided, `cfg.config` is not a dictionary, or config contains unknown keys.
    """
    if kwargs:
        extra = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected keyword arguments to controllers.make(): {extra}.")

    key = str(cfg.name).lower()
    _lazy_ensure(key)
    cls = CONTROLLER_REG[key]
    allowed = _allowed_from_init(cls)

    if not isinstance(cfg.config, dict):
        raise TypeError("ControllerOptions.config must be a dict.")
    unknown = sorted(set(cfg.config) - allowed)
    if unknown:
        allowed_s = ", ".join(sorted(allowed))
        unknown_s = ", ".join(unknown)
        raise TypeError(
            f"Unknown config key(s) for controller '{key}': {unknown_s}. "
            f"Allowed: {allowed_s}.",
        )
    payload: dict[str, Any] = {k: v for k, v in cfg.config.items() if k in allowed}

    return cls(entity=entity, model=model, **payload)
