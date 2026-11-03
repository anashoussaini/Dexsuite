from __future__ import annotations

import threading
import time
from typing import Any, Sequence

try:
    import pysurvive  # type: ignore
except Exception as _e:  # pragma: no cover
    pysurvive = None  # type: ignore[assignment]
    _PYSURVIVE_IMPORT_ERROR: Exception | None = _e
else:  # pragma: no cover
    _PYSURVIVE_IMPORT_ERROR = None


def _require_pysurvive():
    if pysurvive is None:  # pragma: no cover
        raise RuntimeError(
            "Vive device support requires 'pysurvive' and a working libsurvive setup.",
        ) from _PYSURVIVE_IMPORT_ERROR
    return pysurvive


def _decode_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    return text or None


def _float_tuple(values: Any, length: int) -> tuple[float, ...] | None:
    if values is None:
        return None
    try:
        return tuple(float(values[i]) for i in range(length))
    except Exception:
        try:
            seq = list(values)
            if len(seq) < length:
                return None
            return tuple(float(seq[i]) for i in range(length))
        except Exception:
            return None


def _object_name(raw_object: Any) -> str | None:
    try:
        return _decode_text(raw_object.Name())
    except Exception:
        return None


def _object_pose(
    raw_object: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    try:
        pose, _timestamp = raw_object.Pose()
    except Exception:
        return None

    pos = _float_tuple(getattr(pose, "Pos", None), 3)
    quat = _float_tuple(getattr(pose, "Rot", None), 4)
    if pos is None or quat is None:
        return None
    return pos, quat


def _simple_object_ptr(raw_object: Any) -> Any:
    return getattr(raw_object, "ptr", None)


def _simple_object_type(raw_object: Any) -> int | None:
    ps = pysurvive
    ptr = _simple_object_ptr(raw_object)
    if ps is None or ptr is None:
        return None
    try:
        return int(ps.survive_simple_object_get_type(ptr))
    except Exception:
        return None


def _simple_object_subtype(raw_object: Any) -> int | None:
    ps = pysurvive
    ptr = _simple_object_ptr(raw_object)
    if ps is None or ptr is None:
        return None
    try:
        return int(ps.survive_simple_object_get_subtype(ptr))
    except Exception:
        return None


def _simple_button_mask(raw_object: Any) -> int | None:
    ps = pysurvive
    ptr = _simple_object_ptr(raw_object)
    if ps is None or ptr is None:
        return None
    try:
        return int(ps.survive_simple_object_get_button_mask(ptr))
    except Exception:
        return None


def _simple_touch_mask(raw_object: Any) -> int | None:
    ps = pysurvive
    ptr = _simple_object_ptr(raw_object)
    if ps is None or ptr is None:
        return None
    try:
        return int(ps.survive_simple_object_get_touch_mask(ptr))
    except Exception:
        return None


def _simple_axis(raw_object: Any, axis_id: int) -> float | None:
    ps = pysurvive
    ptr = _simple_object_ptr(raw_object)
    if ps is None or ptr is None:
        return None
    try:
        return float(ps.survive_simple_object_get_input_axis(ptr, axis_id))
    except Exception:
        return None


def _mask_has(mask: int | None, button_id: int) -> bool:
    if mask is None:
        return False
    return bool(mask & (1 << int(button_id)))


def _object_is_controller_like(raw_object: Any) -> bool:
    ps = pysurvive
    if ps is None:
        return False

    object_type = _simple_object_type(raw_object)
    if object_type == int(ps.SURVIVE_OBJECT_TYPE_CONTROLLER):
        return True

    subtype = _simple_object_subtype(raw_object)
    return subtype in {
        int(ps.SURVIVE_OBJECT_SUBTYPE_WAND),
        int(ps.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_L),
        int(ps.SURVIVE_OBJECT_SUBTYPE_KNUCKLES_R),
    }


def _object_buttons(raw_object: Any) -> dict[str, Any]:
    ps = pysurvive
    if ps is None or not _object_is_controller_like(raw_object):
        return {}

    button_mask = _simple_button_mask(raw_object)
    touch_mask = _simple_touch_mask(raw_object)

    buttons: dict[str, Any] = {
        "trigger": _simple_axis(raw_object, int(ps.SURVIVE_AXIS_TRIGGER)),
        "trackpad_x": _simple_axis(raw_object, int(ps.SURVIVE_AXIS_TRACKPAD_X)),
        "trackpad_y": _simple_axis(raw_object, int(ps.SURVIVE_AXIS_TRACKPAD_Y)),
        "trackpad_pressed": _mask_has(button_mask, int(ps.SURVIVE_BUTTON_TRACKPAD)),
        "trackpad_touched": _mask_has(touch_mask, int(ps.SURVIVE_BUTTON_TRACKPAD)),
        "grip_button": _mask_has(button_mask, int(ps.SURVIVE_BUTTON_GRIP)),
        "menu_button": _mask_has(button_mask, int(ps.SURVIVE_BUTTON_MENU)),
    }

    thumbstick_x = _simple_axis(raw_object, int(ps.SURVIVE_AXIS_JOYSTICK_X))
    thumbstick_y = _simple_axis(raw_object, int(ps.SURVIVE_AXIS_JOYSTICK_Y))
    if thumbstick_x is not None:
        buttons["thumbstick_x"] = thumbstick_x
        buttons["thumbstick_pressed"] = _mask_has(button_mask, int(ps.SURVIVE_BUTTON_THUMBSTICK))
        buttons["thumbstick_touched"] = _mask_has(touch_mask, int(ps.SURVIVE_BUTTON_THUMBSTICK))
    if thumbstick_y is not None:
        buttons["thumbstick_y"] = thumbstick_y
        buttons.setdefault(
            "thumbstick_pressed",
            _mask_has(button_mask, int(ps.SURVIVE_BUTTON_THUMBSTICK)),
        )
        buttons.setdefault(
            "thumbstick_touched",
            _mask_has(touch_mask, int(ps.SURVIVE_BUTTON_THUMBSTICK)),
        )

    return {key: value for key, value in buttons.items() if value is not None}


class survive_device:
    """Wrapper that matches the pose/button API expected by the teleop code."""

    def __init__(self, ctx: "survive_context", object_name: str, raw_object: Any):
        self.ctx = ctx
        self.object_name = object_name
        self.raw_object = raw_object
        self._pose: tuple[tuple[float, float, float], tuple[float, float, float, float]] | None = None
        self._state_lock = threading.Lock()

    def update_raw_object(self, raw_object: Any) -> None:
        with self._state_lock:
            self.raw_object = raw_object

    def update_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> None:
        with self._state_lock:
            self._pose = pose

    def get_pose_quaternion(self):
        with self._state_lock:
            pose = self._pose
        if pose is None:
            return None
        pos, quat = pose
        return [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]

    def get_serial(self) -> str:
        return self.object_name

    def get_controller_inputs(self) -> dict[str, Any]:
        with self._state_lock:
            raw_object = self.raw_object
        return _object_buttons(raw_object)

    def trigger_haptic_pulse(self, duration_micros: int = 1000, axis_id: int = 0) -> None:
        del duration_micros, axis_id


class survive_context:
    """Minimal registry backed by pysurvive.SimpleContext."""

    def __init__(
        self,
        args: Sequence[str] | None = None,
        discovery_timeout_sec: float = 3.0,
    ):
        ps = _require_pysurvive()
        argv = list(args) if args is not None else [
            "vive_publisher",
            "--no-stdin",
            "--quiet",
        ]

        self.ctx = ps.SimpleContext(argv)
        self.devices: dict[str, survive_device] = {}
        self._backend = "simple"
        self._device_lock = threading.Lock()
        self._closed = threading.Event()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="pysurvive-poll",
            daemon=True,
        )
        self._poll_thread.start()
        self._wait_for_devices(float(discovery_timeout_sec))

    def _iter_raw_objects(self) -> list[Any]:
        try:
            return list(self.ctx.Objects())
        except Exception:
            return []

    def _register_object(self, raw_object: Any) -> survive_device | None:
        object_name = _object_name(raw_object)
        if object_name is None:
            return None

        with self._device_lock:
            dev = self.devices.get(object_name)
            if dev is None:
                dev = survive_device(self, object_name, raw_object)
                self.devices[object_name] = dev
            else:
                dev.update_raw_object(raw_object)
        return dev

    def _sync_object(self, raw_object: Any) -> None:
        dev = self._register_object(raw_object)
        if dev is None:
            return
        pose = _object_pose(raw_object)
        if pose is not None:
            dev.update_pose(pose)

    def _sync_objects(self) -> None:
        for raw_object in self._iter_raw_objects():
            self._sync_object(raw_object)

    def _next_updated(self) -> Any:
        try:
            return self.ctx.NextUpdated()
        except Exception:
            return None

    def _poll_loop(self) -> None:
        while not self._closed.is_set():
            updated = self._next_updated()
            if updated is None:
                self._sync_objects()
                try:
                    if not bool(self.ctx.Running()):
                        break
                except Exception:
                    pass
                time.sleep(0.01)
                continue
            self._sync_object(updated)

    def _wait_for_devices(self, timeout_sec: float) -> None:
        deadline = time.time() + max(timeout_sec, 0.0)
        while time.time() < deadline:
            self._sync_objects()
            if self.devices:
                return
            time.sleep(0.02)
        self._sync_objects()

    def close(self) -> None:
        self._closed.set()
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=0.5)

        ps = pysurvive
        ptr = getattr(self.ctx, "ptr", None)
        close_fn = None if ps is None else getattr(ps, "survive_simple_close", None)
        if ptr is not None and callable(close_fn):
            try:
                close_fn(ptr)
            except Exception:
                pass
