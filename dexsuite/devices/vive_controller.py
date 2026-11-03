"""Vive controller device backends and ZMQ bridge helpers."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import openvr  # type: ignore
except Exception as _e:  # noqa: BLE001  # pragma: no cover
    openvr = None  # type: ignore[assignment]
    _OPENVR_IMPORT_ERROR: Exception | None = _e
else:  # pragma: no cover
    _OPENVR_IMPORT_ERROR = None

from .device import Device
from .survive_utils import survive_context
from .tracker_zmq import tracker_zmq_context, tracker_zmq_publisher
from .triad_utils import (
    TriadLocalPose as tlp,  # noqa: N813
    _require_openvr,
    triad_openvr,
)

DEFAULT_ZMQ_ENDPOINT = "tcp://localhost:5558"
DEFAULT_ZMQ_TOPIC = "lighthouse.poses"
DEFAULT_BRIDGE_ENDPOINT = "tcp://127.0.0.1:5558"
DEFAULT_BRIDGE_TOPIC = "vive.controller"


class BaseViveController(Device):
    """Shared Vive controller action, calibration, button, and status logic."""

    def __init__(
        self,
        controller_id: str = "controller_1",
        side: str = "left",
        *,
        source: str = "auto",
        transport: str,
        endpoint: str | None = None,
        topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        gripper_callback: Callable[[int], None] | None = None,
        haptics: bool = True,
    ) -> None:
        super().__init__(controller="OSCPose_abs_quat", normalized=False, gripper=True)

        self.transport = self._normalize_transport(transport)
        self.discovery_timeout_sec = float(discovery_timeout_sec)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.side = self._normalize_side(side)
        self.controller_id = controller_id
        self.source = self._normalize_source(source)
        self.haptics = haptics
        self._gripper_cb = gripper_callback
        self.endpoint = endpoint
        self.topic = topic

        self.vr, self.vrsystem = self._init_backend(
            endpoint=endpoint,
            topic=topic,
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
        )
        self.dev = self._resolve_controller(self.side, self.controller_id)
        self.name = self._device_name(self.dev)

        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)
        self._has_cal = False

        self._grip_prev = False
        self._trigger_zone = 0

        self.recalibrate()

    @staticmethod
    def _normalize_transport(transport: str) -> str:
        normalized = str(transport).strip().lower()
        if normalized in ("zmq_client", "remote", "bridge"):
            return "zmq"
        return normalized

    @staticmethod
    def _normalize_side(side: str) -> str:
        normalized = str(side).strip().lower()
        if normalized == "any":
            return ""
        return normalized

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source).strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized in ("quat", "quaternion"):
            return "quat"
        if normalized in ("euler", "rpy"):
            return "euler"
        if normalized == "matrix":
            return "matrix"
        raise ValueError("source must be 'auto', 'quat', 'euler', 'rpy', or 'matrix'.")

    @staticmethod
    def _normalize_zmq_source(source: str) -> str:
        """Backward-compatible alias for the old ZMQ client helper."""
        return BaseViveController._normalize_source(source)

    @staticmethod
    def _resolve_alias(
        *,
        value: str | None,
        alias: str | None,
        default: str,
        value_name: str,
        alias_name: str,
    ) -> str:
        if value is not None and alias is not None and value != alias:
            raise ValueError(
                f"Received both {value_name}={value!r} and "
                f"{alias_name}={alias!r}; pass only one value.",
            )
        if value is not None:
            return value
        if alias is not None:
            return alias
        return default

    # -------------------- backend hooks --------------------

    def _init_backend(
        self,
        *,
        endpoint: str | None,
        topic: str | None,
        discovery_timeout_sec: float,
        recv_timeout_ms: int,
        stale_timeout_sec: float,
    ) -> tuple[Any, Any | None]:
        del endpoint, topic, discovery_timeout_sec, recv_timeout_ms, stale_timeout_sec
        raise NotImplementedError

    def _resolve_controller(self, side: str, fallback_id: str):
        del side, fallback_id
        raise NotImplementedError

    # -------------------- device discovery helpers --------------------

    def _device_items(self) -> list[tuple[str, Any]]:
        device_lock = getattr(self.vr, "_device_lock", None)
        if device_lock is None:
            return list(self.vr.devices.items())
        with device_lock:
            return list(self.vr.devices.items())

    @staticmethod
    def _device_serial(dev, fallback: str) -> str:
        return str(getattr(dev, "get_serial", lambda: fallback)())

    def _device_name(self, dev) -> str:
        """Return the transport-specific object name for the controller."""
        for name, registered_dev in self._device_items():
            if registered_dev is dev:
                return name
        return self._device_serial(dev, "controller")

    def _device_matches_side(self, name: str, dev, side: str) -> bool:
        if not side:
            return True
        serial = self._device_serial(dev, name)
        haystacks = (name.lower(), serial.lower())
        return any(side in haystack for haystack in haystacks)

    def _iter_controller_candidates(
        self,
        device_items: list[tuple[str, Any]] | None = None,
    ) -> list[tuple[str, Any]]:
        if device_items is None:
            device_items = self._device_items()

        primary: list[tuple[str, Any]] = []
        secondary: list[tuple[str, Any]] = []
        for name, dev in sorted(device_items):
            label = name.lower()
            if any(
                token in label
                for token in ("hmd", "lighthouse", "base", "tracker")
            ):
                continue
            if label.startswith(("lh0", "lh1", "lh2", "lhb", "tr")):
                continue

            if "controller" in label or label in ("left", "right"):
                primary.append((name, dev))
                continue

            buttons = {}
            with contextlib.suppress(Exception):
                buttons = dev.get_controller_inputs() or {}
            if buttons:
                primary.append((name, dev))
                continue

            if label.startswith(("wm", "ww", "lhr")) or "watchman" in label:
                secondary.append((name, dev))
        return primary + secondary

    def _openvr_controller_role(self, dev) -> str | None:
        if openvr is None:
            return None
        with contextlib.suppress(Exception):
            role = self.vrsystem.getControllerRoleForTrackedDeviceIndex(dev.index)
            if role == openvr.TrackedControllerRole_LeftHand:
                return "left"
            if role == openvr.TrackedControllerRole_RightHand:
                return "right"
        return None

    def _resolve_exact_device(
        self,
        requested: str,
        requested_lower: str,
        device_items: list[tuple[str, Any]],
    ):
        devices = dict(device_items)
        if requested in devices:
            return devices[requested]

        for name, dev in device_items:
            serial = self._device_serial(dev, name)
            if name.lower() == requested_lower or serial.lower() == requested_lower:
                return dev
        return None

    def _resolve_candidate_device(
        self,
        requested_lower: str,
        side: str,
        controller_candidates: list[tuple[str, Any]],
    ):
        if side:
            for name, dev in controller_candidates:
                if self._device_matches_side(name, dev, side):
                    return dev

        if requested_lower.startswith("controller_"):
            suffix = requested_lower.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                idx = int(suffix) - 1
                if 0 <= idx < len(controller_candidates):
                    return controller_candidates[idx][1]

        for name, dev in controller_candidates:
            serial = self._device_serial(dev, name)
            haystacks = (name.lower(), serial.lower())
            if any(requested_lower and requested_lower in haystack for haystack in haystacks):
                return dev

        if controller_candidates:
            return controller_candidates[0][1]
        return None

    def _resolve_registry_controller(self, side: str, fallback_id: str):
        requested = (fallback_id or "").strip()
        requested_lower = requested.lower()
        deadline = time.time() + max(self.discovery_timeout_sec, 0.0)
        last_device_names: list[str] = []

        while True:
            device_items = self._device_items()
            last_device_names = sorted(name for name, _ in device_items)

            dev = self._resolve_exact_device(
                requested,
                requested_lower,
                device_items,
            )
            if dev is not None:
                return dev, last_device_names

            controller_candidates = self._iter_controller_candidates(device_items)
            dev = self._resolve_candidate_device(
                requested_lower,
                side,
                controller_candidates,
            )
            if dev is not None:
                return dev, last_device_names

            if time.time() >= deadline:
                break
            time.sleep(0.05)

        return None, last_device_names

    # -------------------- calibration/status --------------------

    def recalibrate(self) -> None:
        """Resets the (R0, p0) calibration origin."""
        deadline = time.time() + 1.0
        rw_pw: tuple[np.ndarray, np.ndarray] | None = None
        while rw_pw is None and time.time() < deadline:
            rw_pw = tlp.read_pose(self.dev, self.source)
            if rw_pw is None:
                time.sleep(0.002)
        if rw_pw is None:
            return

        rw, pw = rw_pw
        rw = tlp.orthonormalize(rw)

        self.R0 = rw.copy()
        self.p0 = pw.copy()
        self._has_cal = True

        print(f"\n[CONTROLLER] {self.name} {self.side} recalibrated.", flush=True)
        if self.haptics:
            with contextlib.suppress(Exception):
                self.dev.trigger_haptic_pulse(duration_micros=900, axis_id=0)

    def _source_key(self) -> str:
        if self.source in ("auto", "quat"):
            return "quat"
        return self.source

    def _read_source_sample(self):
        if self.source == "matrix":
            return self.dev.get_pose_matrix()
        if self.source == "euler":
            return self.dev.get_pose_euler()
        return self.dev.get_pose_quaternion()

    def pose_source_available(self) -> bool:
        """Return whether the selected pose source currently has a sample."""
        with contextlib.suppress(Exception):
            return self._read_source_sample() is not None
        return False

    @property
    def calibrated(self) -> bool:
        """Return whether a calibration origin has been captured."""
        return self._has_cal

    def _device_status(self, name: str, dev) -> dict[str, Any]:
        serial = name
        with contextlib.suppress(Exception):
            serial = self._device_serial(dev, name)
        status: dict[str, Any] = {
            "name": name,
            "serial": str(serial),
            "selected": dev is self.dev,
        }
        pose_status = getattr(dev, "pose_status", None)
        if callable(pose_status):
            with contextlib.suppress(Exception):
                status.update(pose_status())
                return status

        for key, method_name in (
            ("quat", "get_pose_quaternion"),
            ("euler", "get_pose_euler"),
            ("matrix", "get_pose_matrix"),
        ):
            sample = None
            with contextlib.suppress(Exception):
                sample = getattr(dev, method_name)()
            status[f"readable_{key}"] = sample is not None
            status[f"raw_{key}"] = sample is not None
        status["fresh"] = any(
            bool(status.get(f"readable_{key}"))
            for key in ("quat", "euler", "matrix")
        )
        status["age_sec"] = None
        return status

    def bridge_status(self) -> dict[str, Any]:
        """Return diagnostic information for the active controller transport."""
        devices = [self._device_status(name, dev) for name, dev in self._device_items()]
        selected = next((device for device in devices if device.get("selected")), None)
        source_key = self._source_key()
        return {
            "transport": self.transport,
            "endpoint": self.endpoint,
            "topic": self.topic,
            "side": self.side,
            "controller_id": self.controller_id,
            "selected_device": self.name,
            "source": self.source,
            "source_key": source_key,
            "calibrated": self._has_cal,
            "source_available": bool(
                selected and selected.get(f"readable_{source_key}"),
            ),
            "devices": devices,
        }

    def bridge_status_text(self) -> str:
        """Return a compact human-readable bridge status string."""
        status = self.bridge_status()
        device_parts = []
        for device in status["devices"]:
            raw = ",".join(
                key
                for key in ("quat", "euler", "matrix")
                if device.get(f"raw_{key}")
            )
            readable = ",".join(
                key
                for key in ("quat", "euler", "matrix")
                if device.get(f"readable_{key}")
            )
            age_sec = device.get("age_sec")
            age = "unknown" if age_sec is None else f"{float(age_sec):.3f}s"
            selected = "*" if device.get("selected") else ""
            device_parts.append(
                f"{selected}{device['name']}(age={age}, "
                f"raw={raw or 'none'}, readable={readable or 'none'})",
            )
        devices = "; ".join(device_parts) or "none"
        return (
            f"transport={status['transport']} endpoint={status['endpoint']} "
            f"topic={status['topic']} selected={status['selected_device']} "
            f"source={status['source']} calibrated={status['calibrated']} "
            f"source_available={status['source_available']} devices=[{devices}]"
        )

    # -------------------- inputs -> actions --------------------

    def _parse_actions(self, buttons: dict[str, Any] | None) -> list[int]:
        if not buttons:
            return [0]

        grip_pressed = bool(buttons.get("grip_button", False) or buttons.get("grip", False))
        if (not self._grip_prev) and grip_pressed:
            self.recalibrate()
        self._grip_prev = grip_pressed

        trig = float(buttons.get("trigger", 0.0))
        new_zone = 1 if trig > 0.90 else (-1 if trig < 0.10 else 0)

        action = 0
        if new_zone != self._trigger_zone:
            self._trigger_zone = new_zone
            if new_zone in (-1, 1):
                action = new_zone
                if self._gripper_cb is not None:
                    with contextlib.suppress(Exception):
                        self._gripper_cb(action)
                if self.haptics:
                    with contextlib.suppress(Exception):
                        self.dev.trigger_haptic_pulse(duration_micros=600, axis_id=0)
        return [action]

    def _current_buttons(self) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            return self.dev.get_controller_inputs() or {}
        return {}

    def _empty_action(self, buttons: dict[str, Any] | None = None) -> dict[str, Any]:
        button_state = {} if buttons is None else dict(buttons)
        gripper_action = self._parse_actions(button_state) if self.gripper else [0]
        return {
            "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "buttons": button_state,
            "gripper": gripper_action,
            "meta": {
                "id": self.name,
                "side": self.side,
                "transport": self.transport,
                "t": time.time(),
            },
        }

    # -------------------- main read --------------------

    def read(self) -> dict[str, Any]:
        """Read the latest controller pose, buttons, and gripper action."""
        if not self._has_cal and self.pose_source_available():
            self.recalibrate()

        buttons = self._current_buttons()
        if not self._has_cal:
            return self._empty_action(buttons)

        rw_pw = tlp.read_pose(self.dev, self.source)
        if rw_pw is None:
            return self._empty_action(buttons)

        rw, pw = rw_pw
        rw = tlp.orthonormalize(rw)

        rloc = self.R0.T @ rw
        ploc = self.R0.T @ (pw - self.p0)

        axes = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
        rloc = axes @ rloc @ axes.T
        ploc = axes @ ploc

        qw, qx, qy, qz = tlp.quat_wxyz_from_R(rloc)
        gripper_action = self._parse_actions(buttons) if self.gripper else [0]

        return {
            "pose": [
                float(ploc[0]),
                float(ploc[1]),
                float(ploc[2]),
                qw,
                qx,
                qy,
                qz,
            ],
            "buttons": buttons,
            "gripper": gripper_action,
            "meta": {
                "id": self.name,
                "side": self.side,
                "transport": self.transport,
                "t": time.time(),
            },
        }

    def get_action(self):
        """Return the latest controller action dictionary."""
        return self.read()

    def close(self) -> None:
        """Close the underlying transport when it exposes a close method."""
        with contextlib.suppress(Exception):
            self.vr.close()


class ZMQViveController(BaseViveController):
    """Vive controller reader backed by controller ZMQ snapshots."""

    def __init__(
        self,
        controller_id: str = "controller_1",
        side: str = "left",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        gripper_callback: Callable[[int], None] | None = None,
        haptics: bool = True,
    ) -> None:
        endpoint = self._resolve_alias(
            value=endpoint,
            alias=zmq_endpoint,
            default=DEFAULT_ZMQ_ENDPOINT,
            value_name="endpoint",
            alias_name="zmq_endpoint",
        )
        topic = self._resolve_alias(
            value=topic,
            alias=zmq_topic,
            default=DEFAULT_ZMQ_TOPIC,
            value_name="topic",
            alias_name="zmq_topic",
        )
        self.zmq_endpoint = endpoint
        self.zmq_topic = topic
        super().__init__(
            controller_id=controller_id,
            side=side,
            source=source,
            transport="zmq",
            endpoint=endpoint,
            topic=topic,
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            gripper_callback=gripper_callback,
            haptics=haptics,
        )

    def _init_backend(
        self,
        *,
        endpoint: str | None,
        topic: str | None,
        discovery_timeout_sec: float,
        recv_timeout_ms: int,
        stale_timeout_sec: float,
    ) -> tuple[Any, None]:
        return (
            tracker_zmq_context(
                endpoint=endpoint or DEFAULT_ZMQ_ENDPOINT,
                topic=topic or DEFAULT_ZMQ_TOPIC,
                discovery_timeout_sec=discovery_timeout_sec,
                recv_timeout_ms=recv_timeout_ms,
                stale_timeout_sec=stale_timeout_sec,
            ),
            None,
        )

    def _resolve_controller(self, side: str, fallback_id: str):
        dev, last_device_names = self._resolve_registry_controller(side, fallback_id)
        if dev is not None:
            return dev
        raise RuntimeError(
            f"No Vive controller samples received via ZMQ from {self.endpoint} "
            f"[{self.topic}]. Received devices: {last_device_names!r}. "
            "Is ViveControllerZMQServer or another compatible publisher "
            "running with the same topic?",
        )


class SurviveViveController(BaseViveController):
    """Vive controller reader backed by libsurvive/pysurvive."""

    def __init__(
        self,
        controller_id: str = "controller_1",
        side: str = "left",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        gripper_callback: Callable[[int], None] | None = None,
        haptics: bool = True,
    ) -> None:
        del endpoint, topic, zmq_endpoint, zmq_topic
        super().__init__(
            controller_id=controller_id,
            side=side,
            source=source,
            transport="survive",
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            gripper_callback=gripper_callback,
            haptics=haptics,
        )

    def _init_backend(
        self,
        *,
        endpoint: str | None,
        topic: str | None,
        discovery_timeout_sec: float,
        recv_timeout_ms: int,
        stale_timeout_sec: float,
    ) -> tuple[Any, None]:
        del endpoint, topic, recv_timeout_ms, stale_timeout_sec
        return survive_context(discovery_timeout_sec=discovery_timeout_sec), None

    def _resolve_controller(self, side: str, fallback_id: str):
        dev, _last_device_names = self._resolve_registry_controller(side, fallback_id)
        if dev is not None:
            return dev
        raise RuntimeError(
            "No controller-like devices received via libsurvive. Are base stations on?",
        )


class OpenVRViveController(BaseViveController):
    """Vive controller reader backed by SteamVR/OpenVR."""

    def __init__(
        self,
        controller_id: str = "controller_1",
        side: str = "left",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        gripper_callback: Callable[[int], None] | None = None,
        haptics: bool = True,
    ) -> None:
        del endpoint, topic, zmq_endpoint, zmq_topic
        super().__init__(
            controller_id=controller_id,
            side=side,
            source=source,
            transport="openvr",
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            gripper_callback=gripper_callback,
            haptics=haptics,
        )

    def _init_backend(
        self,
        *,
        endpoint: str | None,
        topic: str | None,
        discovery_timeout_sec: float,
        recv_timeout_ms: int,
        stale_timeout_sec: float,
    ) -> tuple[Any, Any]:
        del endpoint, topic, discovery_timeout_sec, recv_timeout_ms, stale_timeout_sec
        if openvr is None:
            raise RuntimeError(
                "Vive controller support requires 'openvr' (pip install openvr) "
                "and a working SteamVR installation.",
            ) from _OPENVR_IMPORT_ERROR
        vr = triad_openvr()
        return vr, vr.vrsystem

    def _resolve_controller(self, side: str, fallback_id: str):
        if openvr is None:
            raise RuntimeError(
                "Vive controller support requires 'openvr' (pip install openvr) "
                "and a working SteamVR installation.",
            ) from _OPENVR_IMPORT_ERROR

        if side:
            for dev in self.vr.devices.values():
                if getattr(dev, "device_class", "") != "Controller":
                    continue
                if self._openvr_controller_role(dev) == side:
                    return dev

        if fallback_id in self.vr.devices:
            return self.vr.devices[fallback_id]
        for dev in self.vr.devices.values():
            if getattr(dev, "device_class", "") == "Controller":
                return dev
        raise RuntimeError("No Vive controllers found. Is SteamVR running?")


def _controller_class_for_transport(transport: str) -> type[BaseViveController]:
    normalized = BaseViveController._normalize_transport(transport)
    controller_classes: dict[str, type[BaseViveController]] = {
        "openvr": OpenVRViveController,
        "survive": SurviveViveController,
        "zmq": ZMQViveController,
    }
    try:
        return controller_classes[normalized]
    except KeyError:
        raise ValueError(f"Unsupported controller transport: {transport!r}") from None


def ViveController(
    controller_id: str = "controller_1",
    side: str = "left",
    *,
    source: str = "auto",
    transport: str = "zmq",
    endpoint: str | None = None,
    topic: str | None = None,
    zmq_endpoint: str | None = None,
    zmq_topic: str | None = None,
    discovery_timeout_sec: float = 3.0,
    recv_timeout_ms: int = 100,
    stale_timeout_sec: float = 0.5,
    gripper_callback: Callable[[int], None] | None = None,
    haptics: bool = True,
) -> BaseViveController:
    """Create a Vive controller using the requested backend transport."""
    controller_cls = _controller_class_for_transport(transport)
    return controller_cls(
        controller_id=controller_id,
        side=side,
        source=source,
        endpoint=endpoint,
        topic=topic,
        zmq_endpoint=zmq_endpoint,
        zmq_topic=zmq_topic,
        discovery_timeout_sec=discovery_timeout_sec,
        recv_timeout_ms=recv_timeout_ms,
        stale_timeout_sec=stale_timeout_sec,
        gripper_callback=gripper_callback,
        haptics=haptics,
    )


class ViveControllerZMQServer:
    """Publish raw SteamVR/OpenVR Vive controller samples over ZMQ."""

    def __init__(
        self,
        controller_id: str = "controller_1",
        side: str = "left",
        *,
        bind_endpoint: str = "tcp://*:5558",
        topic: str = DEFAULT_BRIDGE_TOPIC,
        rate_hz: float = 120.0,
        publish_all: bool = False,
        poll_events: bool = True,
    ) -> None:
        """Initialize the SteamVR-side publisher."""
        self.vr = triad_openvr()
        self.vrsystem = self.vr.vrsystem
        self.publisher = tracker_zmq_publisher(
            bind_endpoint=bind_endpoint,
            topic=topic,
        )
        self.controller_id = controller_id
        self.side = BaseViveController._normalize_side(side)
        self.bind_endpoint = bind_endpoint
        self.topic = topic
        self.rate_hz = float(rate_hz)
        self.publish_all = bool(publish_all)
        self.poll_events = bool(poll_events)
        self._selected_dev = None if self.publish_all else self._resolve_controller()
        self._last_published_device_names: tuple[str, ...] = ()
        self._last_published_device_formats: dict[str, tuple[str, ...]] = {}

    def _controller_role(self, dev) -> str | None:
        ov = _require_openvr()
        with contextlib.suppress(Exception):
            role = self.vrsystem.getControllerRoleForTrackedDeviceIndex(dev.index)
            if role == ov.TrackedControllerRole_LeftHand:
                return "left"
            if role == ov.TrackedControllerRole_RightHand:
                return "right"
        return None

    def _controller_items(self) -> list[tuple[str, Any]]:
        return [
            (name, dev)
            for name, dev in sorted(self.vr.devices.items())
            if getattr(dev, "device_class", "") == "Controller"
        ]

    def _resolve_controller(self):
        requested = (self.controller_id or "").strip()
        requested_lower = requested.lower()

        if self.side:
            for _, dev in self._controller_items():
                if self._controller_role(dev) == self.side:
                    return dev
            names = [name for name, _ in self._controller_items()]
            raise RuntimeError(
                f"No Vive controller with OpenVR role {self.side!r} found. "
                f"Discovered controllers: {names!r}. Use side='any' with an "
                "explicit controller_id if SteamVR roles are unavailable.",
            )

        if requested in self.vr.devices:
            dev = self.vr.devices[requested]
            if getattr(dev, "device_class", "") == "Controller":
                return dev

        for name, dev in self._controller_items():
            serial = BaseViveController._device_serial(dev, name)
            if name.lower() == requested_lower or serial.lower() == requested_lower:
                return dev

        names = [name for name, _ in self._controller_items()]
        raise RuntimeError(
            f"No Vive controller matching controller_id={requested!r} found. "
            f"Discovered controllers: {names!r}.",
        )

    def _publish_name(self, name: str, dev) -> str:
        role = self._controller_role(dev)
        if role:
            return f"controller_{role}"
        return name

    def _publish_devices(self) -> dict[str, Any]:
        if self.publish_all:
            return {
                self._publish_name(name, dev): dev
                for name, dev in self._controller_items()
            }

        selected = self._selected_dev
        if selected is None:
            return {}
        for name, dev in self._controller_items():
            if dev is selected:
                return {self._publish_name(name, dev): dev}
        return {self._publish_name(self.controller_id, selected): selected}

    def publish_once(self) -> int:
        """Publish one controller snapshot and return the number of devices sent."""
        if self.poll_events:
            with contextlib.suppress(Exception):
                self.vr.poll_vr_events()
        devices = self._publish_devices()
        self._last_published_device_names = tuple(sorted(str(name) for name in devices))
        count = self.publisher.publish_snapshot(devices)
        self._last_published_device_formats = self.publisher.last_device_formats
        return count

    @property
    def last_published_device_names(self) -> tuple[str, ...]:
        """Return the device names included in the most recent snapshot."""
        return self._last_published_device_names

    @property
    def last_published_device_formats(self) -> dict[str, tuple[str, ...]]:
        """Return pose/button formats included in the most recent snapshot."""
        return dict(self._last_published_device_formats)

    def run_forever(self, rate_hz: float | None = None) -> None:
        """Publish controller samples until interrupted."""
        hz = self.rate_hz if rate_hz is None else float(rate_hz)
        period_sec = 0.0 if hz <= 0.0 else 1.0 / hz
        try:
            while True:
                self.publish_once()
                if period_sec > 0.0:
                    time.sleep(period_sec)
        except KeyboardInterrupt:
            return

    serve_forever = run_forever

    def close(self) -> None:
        """Close the ZMQ publisher backing this server."""
        with contextlib.suppress(Exception):
            self.publisher.close()


__all__ = [
    "BaseViveController",
    "OpenVRViveController",
    "SurviveViveController",
    "ViveController",
    "ViveControllerZMQServer",
    "ZMQViveController",
]
