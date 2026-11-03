"""Build DexSuite environments from task names and configuration options.

Environment creation accepts either direct keyword options or options objects.
Genesis is initialized lazily on the global device, and construction temporarily
uses a real stderr stream when notebook streams do not expose file descriptors.
"""

from __future__ import annotations

import contextlib
import inspect
import sys

from dexsuite.core.registry import ENV_REG
from dexsuite.options.builders import build_make_options, normalize_make_options
from dexsuite.utils import get_device

_GENESIS_STATE = {"ready": False}


def _has_fileno(stream: object) -> bool:
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return False
    else:
        return True


@contextlib.contextmanager
def _ensure_stderr_fileno() -> None:
    """Use a real stderr while importing or constructing Genesis objects.

    Genesis may access the file descriptor behind stderr during initialization.
    In notebook environments this context temporarily swaps in `sys.__stderr__`
    when available, then restores the original stream before returning.
    """
    if _has_fileno(sys.stderr):
        yield
        return

    original = sys.stderr
    fallback = getattr(sys, "__stderr__", None)
    if fallback is not None and _has_fileno(fallback):
        sys.stderr = fallback
        try:
            yield
        finally:
            sys.stderr = original
        return

    yield


# Genesis may redirect libc stderr at import-time, which fails in notebooks
# (ipykernel stderr does not implement fileno()).
with _ensure_stderr_fileno():
    import genesis as gs


def _lazy_init_genesis(*, performance_mode: bool, precision: str = "32") -> None:
    if _GENESIS_STATE["ready"]:
        if precision != _GENESIS_STATE.get("precision", "32"):
            raise RuntimeError("Genesis precision cannot change within a process; start a new worker.")
        return
    dev = get_device()
    gs.init(
        backend=gs.gpu if dev.type == "cuda" else gs.cpu,
        logging_level="error",
        performance_mode=performance_mode,
        theme="dark",
        precision=precision,
    )
    _GENESIS_STATE["ready"] = True
    _GENESIS_STATE["precision"] = precision


__all__ = ["make"]


def make(task: str, /, **kw):  # noqa: C901
    """Create a registered DexSuite environment.

    Direct keyword options are resolved into option objects before the
    environment is constructed. Unknown task names fail before Genesis
    initialization, while unsupported environment constructor keywords fail with
    a `TypeError`.

    Args:
        task: Registered task name to instantiate.
        kw: Direct options or options objects accepted by DexSuite builders.

    Returns:
        Constructed environment instance for the requested task.

    Raises:
        TypeError: When `task` is not a string or the selected environment does not accept a requested keyword.
        ValueError: When `task` is not present in the environment registry.
    """
    kw = normalize_make_options(kw)
    if not isinstance(task, str):
        raise TypeError("task must be a string.")
    env_cls = ENV_REG.get(task.lower())
    if env_cls is None:
        available = ", ".join(sorted(ENV_REG)) or "<none>"
        raise ValueError(
            f"Unknown task '{task}'. Available tasks: {available}.",
        )
    robot, cameras, sim, render_mode = build_make_options(kw)
    _lazy_init_genesis(performance_mode=sim.performance_mode, precision=sim.precision)

    # Introspect env signature
    sig = inspect.signature(env_cls)
    params = sig.parameters
    accepted = set(params.keys())
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    call: dict[str, object] = {}

    # Pass resolved options objects to environments that accept them.
    if "robot" in accepted:
        call["robot"] = robot
    if "cameras" in accepted:
        call["cameras"] = cameras
    if "sim" in accepted:
        call["sim"] = sim
    if "render_mode" in accepted:
        call["render_mode"] = render_mode
    if "camera_gui" in kw:
        if "camera_gui" not in accepted and not accepts_kwargs:
            raise TypeError(f"{env_cls.__name__} does not accept camera_gui.")
        call["camera_gui"] = kw["camera_gui"]

    # If an environment accepts explicit timing knobs, derive them from sim.
    if "sim_dt" in accepted and "sim_dt" not in call:
        call["sim_dt"] = 1.0 / float(sim.control_hz)
    if "substeps" in accepted and "substeps" not in call:
        call["substeps"] = 1
    if "horizon" in accepted and "horizon" not in call:
        call["horizon"] = 200

    with _ensure_stderr_fileno():
        return env_cls(**call)
