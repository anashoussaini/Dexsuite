from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import numpy as np

from .triad_utils import TriadLocalPose as tlp

try:
    import zmq  # type: ignore
except Exception as _e:  # pragma: no cover
    zmq = None  # type: ignore[assignment]
    _ZMQ_IMPORT_ERROR: Exception | None = _e
else:  # pragma: no cover
    _ZMQ_IMPORT_ERROR = None


def _require_zmq():
    if zmq is None:  # pragma: no cover
        raise RuntimeError(
            "Tracker ZMQ transport requires 'pyzmq' (pip install pyzmq).",
        ) from _ZMQ_IMPORT_ERROR
    return zmq


def _coerce_pose(payload: Any) -> list[float] | None:
    if not isinstance(payload, (list, tuple)) or len(payload) < 7:
        return None
    try:
        return [float(v) for v in payload[:7]]
    except (TypeError, ValueError):
        return None


def _coerce_buttons(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    coerced: dict[str, Any] = {}
    for key, value in payload.items():
        normalized = _coerce_json_value(value)
        if normalized is None:
            continue
        coerced[str(key)] = normalized
    return coerced


def _coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "item") and callable(value.item):
        try:
            return _coerce_json_value(value.item())
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_coerce_json_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class tracker_zmq_device:
    """Tracked device backed by the latest ZMQ sample."""

    def __init__(
        self,
        ctx: "tracker_zmq_context",
        object_name: str,
        stale_timeout_sec: float,
    ):
        self.ctx = ctx
        self.object_name = object_name
        self._pose: list[float] | None = None
        self._buttons: dict[str, Any] = {}
        self._last_update_t = 0.0
        self._stale_timeout_sec = float(stale_timeout_sec)
        self._pose_lock = threading.Lock()

    def _latest_pose(self) -> tuple[list[float] | None, float | None, bool]:
        with self._pose_lock:
            pose = None if self._pose is None else list(self._pose)
            last_update_t = self._last_update_t
        if pose is None:
            return None, None, False
        age_sec = time.time() - last_update_t
        return pose, age_sec, age_sec <= self._stale_timeout_sec

    def update_state(
        self,
        pose: list[float] | None,
        buttons: dict[str, Any],
        sample_time: float,
    ) -> None:
        with self._pose_lock:
            if pose is not None:
                self._pose = list(pose)
            self._buttons = dict(buttons)
            self._last_update_t = float(sample_time)

    def get_pose_quaternion(self):
        pose, _age_sec, fresh = self._latest_pose()
        if pose is None or not fresh:
            return None
        return pose

    def get_pose_matrix(self):
        pose = self.get_pose_quaternion()
        if pose is None:
            return None
        x, y, z, qw, qx, qy, qz = pose[:7]
        R = tlp.R_from_quat_wxyz(qw, qx, qy, qz)
        M = np.zeros((3, 4), dtype=float)
        M[:, :3] = R
        M[:, 3] = [x, y, z]
        return M

    def get_pose_euler(self):
        pose = self.get_pose_quaternion()
        if pose is None:
            return None
        x, y, z, qw, qx, qy, qz = pose[:7]
        R = tlp.R_from_quat_wxyz(qw, qx, qy, qz)
        roll, pitch, yaw = tlp.euler_rpy_from_R_zyx(R)
        return [
            x,
            y,
            z,
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        ]

    def pose_status(self) -> dict[str, Any]:
        pose, age_sec, fresh = self._latest_pose()
        raw_quat = pose is not None
        status: dict[str, Any] = {
            "age_sec": age_sec,
            "fresh": bool(raw_quat and fresh),
        }
        for key in ("quat", "euler", "matrix"):
            status[f"raw_{key}"] = raw_quat
            status[f"readable_{key}"] = bool(raw_quat and fresh)
        return status

    def get_serial(self) -> str:
        return self.object_name

    def get_controller_inputs(self) -> dict[str, Any]:
        with self._pose_lock:
            return dict(self._buttons)

    def trigger_haptic_pulse(self, duration_micros: int = 1000, axis_id: int = 0) -> None:
        del duration_micros, axis_id


class tracker_zmq_context:
    """Subscriber-side registry that mirrors tracked devices from a PUB socket."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5557",
        topic: str = "lighthouse.poses",
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
    ):
        zq = _require_zmq()
        self.endpoint = endpoint
        self.topic = str(topic)
        self.devices: dict[str, tracker_zmq_device] = {}

        self._device_lock = threading.Lock()
        self._closed = threading.Event()
        self._recv_timeout_ms = int(recv_timeout_ms)
        self._stale_timeout_sec = float(stale_timeout_sec)

        self._ctx = zq.Context.instance()
        self._socket = self._ctx.socket(zq.SUB)
        self._socket.setsockopt(zq.LINGER, 0)
        self._socket.setsockopt(zq.RCVHWM, 1)
        self._socket.setsockopt(zq.SUBSCRIBE, self.topic.encode("utf-8"))
        self._socket.connect(self.endpoint)

        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name="tracker-zmq-sub",
            daemon=True,
        )
        self._recv_thread.start()
        self._wait_for_devices(float(discovery_timeout_sec))

    def _recv_loop(self) -> None:
        zq = _require_zmq()
        poller = zq.Poller()
        poller.register(self._socket, zq.POLLIN)

        while not self._closed.is_set():
            try:
                events = dict(poller.poll(self._recv_timeout_ms))
            except Exception:
                if self._closed.is_set():
                    break
                continue
            if self._socket not in events:
                continue

            try:
                topic, payload = self._socket.recv_multipart(flags=zq.NOBLOCK)
            except zq.Again:
                continue
            except Exception:
                if self._closed.is_set():
                    break
                continue

            if topic.decode("utf-8", errors="ignore") != self.topic:
                continue

            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception:
                continue

            self._apply_message(message)

    def _apply_message(self, message: dict[str, Any]) -> None:
        receive_time = time.time()
        devices = message.get("devices", {})
        if not isinstance(devices, dict):
            return

        with self._device_lock:
            for object_name, state in devices.items():
                if not isinstance(object_name, str) or not isinstance(state, dict):
                    continue
                pose = _coerce_pose(state.get("pose"))
                buttons = _coerce_buttons(state.get("buttons"))
                if pose is None and not buttons:
                    continue

                dev = self.devices.get(object_name)
                if dev is None:
                    dev = tracker_zmq_device(
                        self,
                        object_name,
                        stale_timeout_sec=self._stale_timeout_sec,
                    )
                    self.devices[object_name] = dev
                dev.update_state(
                    pose=pose,
                    buttons=buttons,
                    sample_time=receive_time,
                )

    def _wait_for_devices(self, timeout_sec: float) -> None:
        deadline = time.time() + max(timeout_sec, 0.0)
        while time.time() < deadline:
            with self._device_lock:
                if self.devices:
                    return
            time.sleep(0.02)

    def close(self) -> None:
        self._closed.set()
        recv_thread = getattr(self, "_recv_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            recv_thread.join(timeout=max(0.2, self._recv_timeout_ms / 1000.0 + 0.1))
        try:
            self._socket.close(0)
        except Exception:
            pass


class tracker_zmq_publisher:
    """Publisher that broadcasts the latest pose snapshot for all tracked devices."""

    def __init__(
        self,
        bind_endpoint: str = "tcp://*:5557",
        topic: str = "lighthouse.poses",
    ):
        zq = _require_zmq()
        self.topic = str(topic)
        self._ctx = zq.Context.instance()
        self._socket = self._ctx.socket(zq.PUB)
        self._socket.setsockopt(zq.LINGER, 0)
        self._socket.setsockopt(zq.SNDHWM, 1)
        self._socket.bind(bind_endpoint)
        self.last_device_formats: dict[str, tuple[str, ...]] = {}

    def publish_snapshot(self, devices: dict[str, Any], published_at: float | None = None) -> int:
        sent_at = time.time() if published_at is None else float(published_at)
        payload: dict[str, Any] = {"t": sent_at, "devices": {}}
        device_formats: dict[str, tuple[str, ...]] = {}

        for object_name, dev in sorted(devices.items()):
            try:
                pose = dev.get_pose_quaternion()
            except Exception:
                continue
            try:
                buttons = dev.get_controller_inputs() or {}
            except Exception:
                buttons = {}
            pose = _coerce_pose(pose)
            buttons = _coerce_buttons(buttons)
            if pose is None and not buttons:
                continue
            state: dict[str, Any] = {
                "serial": getattr(dev, "get_serial", lambda: object_name)(),
                "t": sent_at,
            }
            if pose is not None:
                state["pose"] = pose
            if buttons:
                state["buttons"] = buttons
            payload["devices"][str(object_name)] = state
            formats = []
            if pose is not None:
                formats.append("quat")
            if buttons:
                formats.append("buttons")
            device_formats[str(object_name)] = tuple(formats)

        self._socket.send_multipart(
            [
                self.topic.encode("utf-8"),
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            ],
        )
        self.last_device_formats = device_formats
        return len(payload["devices"])

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception:
            pass
