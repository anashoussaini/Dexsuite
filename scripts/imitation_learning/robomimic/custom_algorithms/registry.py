"""Register DexSuite's own robomimic algorithms with robomimic's registries."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def register_custom_algorithms() -> None:
    """Import every module of this package so robomimic knows its algorithms.

    The package ships its own ACT (act.py, act_config.py, act_nets.py), the code
    behind the DexSuite paper's ACT results; robomimic 0.5.0 has no ACT. robomimic's
    registries are plain dicts, so if a robomimic release brings its own "act", the
    local one replaces it here, once and with a message, and robomimic's ACT
    modules are never imported by this package.
    """
    before = _registered_algorithms()
    _import_local_modules()
    for name, entry in _registered_algorithms().items():
        previous = before.get(name)
        if previous is not None and previous != entry and not _is_local(previous):
            print(f"[robomimic] '{name}' is DexSuite's own {name}, not robomimic's")


def _registered_algorithms() -> dict[str, tuple[object, object]]:
    from robomimic.algo.algo import REGISTERED_ALGO_FACTORY_FUNCS
    from robomimic.config.base_config import REGISTERED_CONFIGS

    names = set(REGISTERED_CONFIGS) | set(REGISTERED_ALGO_FACTORY_FUNCS)
    return {
        name: (REGISTERED_CONFIGS.get(name), REGISTERED_ALGO_FACTORY_FUNCS.get(name))
        for name in names
    }


def _is_local(entry: tuple[object, object]) -> bool:
    return all(
        obj is None or str(getattr(obj, "__module__", "")).startswith(f"{__package__}.")
        for obj in entry
    )


def _import_local_modules() -> None:
    package = __package__
    package_dir = Path(__file__).resolve().parent
    for module in pkgutil.iter_modules([str(package_dir)]):
        if module.name in {"registry"} or module.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{module.name}")
