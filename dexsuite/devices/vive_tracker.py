"""Vive tracker device backends and ZMQ bridge helpers."""

from __future__ import annotations

import contextlib
import time
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
DEFAULT_BRIDGE_ENDPOINT = "tcp://127.0.0.1:5559"
DEFAULT_BRIDGE_TOPIC = "vive.tracker"


class BaseViveTracker(Device):
    """Shared Vive tracker action, calibration, and status logic."""

    def __init__(
        self,
        tracker_id: str = "tracker_1",
        side: str = "center",
        *,
        source: str = "auto",
        transport: str,
        endpoint: str | None = None,
        topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        haptics: bool = True,
        gripper: bool = True,
        reset_key: str = "r",
    ) -> None:
        super().__init__(
            controller="OSCPOSE_abs_quat",
            normalized=False,
            gripper=gripper,
        )

        self.side = str(side).lower()
        self.tracker_id = tracker_id
        self.source = self._normalize_source(source)
        self.transport = self._normalize_transport(transport)
        self.discovery_timeout_sec = float(discovery_timeout_sec)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.haptics = haptics
        self.endpoint = endpoint
        self.topic = topic
        del reset_key

        self.vr, self.vrsystem = self._init_backend(
            endpoint=endpoint,
            topic=topic,
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
        )
        self.dev = self._resolve_tracker(self.tracker_id)
        self.name = self._device_name(self.dev)

        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)
        self._has_cal = False

        self.recalibrate()

    @staticmethod
    def _normalize_transport(transport: str) -> str:
        normalized = str(transport).strip().lower()
        if normalized in ("zmq_client", "remote", "bridge"):
            return "zmq"
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
        return BaseViveTracker._normalize_source(source)

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

    def _resolve_tracker(self, fallback_id: str):
        del fallback_id
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
        """Return the transport-specific object name for the tracker."""
        if self.transport == "openvr":
            for name, registered_dev in self.vr.devices.items():
                if getattr(registered_dev, "index", None) == getattr(dev, "index", None):
                    return name
            return f"tracker@{getattr(dev, 'index', '?')}"

        for name, registered_dev in self._device_items():
            if registered_dev is dev:
                return name
        return self._device_serial(dev, "tracker")

    def _device_matches_side(self, name: str, dev, side: str) -> bool:
        if not side:
            return True
        serial = self._device_serial(dev, name)
        haystacks = (name.lower(), str(serial).lower())
        return any(side in haystack for haystack in haystacks)

    def _iter_tracker_candidates(
        self,
        device_items: list[tuple[str, Any]] | None = None,
    ) -> list[tuple[str, Any]]:
        primary: list[tuple[str, Any]] = []
        secondary: list[tuple[str, Any]] = []
        if device_items is None:
            device_items = self._device_items()
        for name, dev in sorted(device_items):
            label = name.lower()
            if any(
                token in label
                for token in ("hmd", "lighthouse", "base", "controller")
            ):
                continue
            if label.startswith(("lh0", "lh1", "lh2", "lhb", "controller")):
                continue
            if label.startswith("tr") or "tracker" in label:
                primary.append((name, dev))
                continue
            if label.startswith(("wm", "ww", "lhr")) or "watchman" in label:
                secondary.append((name, dev))
        return primary + secondary

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
        side_lower: str,
        tracker_candidates: list[tuple[str, Any]],
    ):
        if side_lower:
            for name, dev in tracker_candidates:
                if self._device_matches_side(name, dev, side_lower):
                    return dev

        if requested_lower.startswith("tracker_"):
            suffix = requested_lower.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                idx = int(suffix) - 1
                if 0 <= idx < len(tracker_candidates):
                    return tracker_candidates[idx][1]

        for name, dev in tracker_candidates:
            serial = self._device_serial(dev, name)
            haystacks = (name.lower(), serial.lower())
            if any(requested_lower and requested_lower in haystack for haystack in haystacks):
                return dev

        if tracker_candidates:
            return tracker_candidates[0][1]
        return None

    def _resolve_registry_tracker(self, fallback_id: str):
        requested = (fallback_id or "").strip()
        requested_lower = requested.lower()
        side_lower = (self.side or "").lower()
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

            tracker_candidates = self._iter_tracker_candidates(device_items)
            dev = self._resolve_candidate_device(
                requested_lower,
                side_lower,
                tracker_candidates,
            )
            if dev is not None:
                return dev, last_device_names

            if time.time() >= deadline:
                break
            time.sleep(0.05)

        return None, last_device_names

    # -------------------- calibration/status --------------------

    def _set_calibration(self, Rw: np.ndarray, pw: np.ndarray) -> None:  # noqa: N803
        """Internal helper to set the (R0, p0) calibration pose."""
        rw = tlp.orthonormalize(Rw)
        self.R0 = rw.copy()
        self.p0 = pw.copy()
        self._has_cal = True
        print(f"\n[TRACKER] {self.name} {self.side} recalibrated.", flush=True)
        if self.haptics:
            with contextlib.suppress(Exception):
                self.dev.trigger_haptic_pulse(duration_micros=900, axis_id=0)

    def recalibrate(
        self,
        Rw: np.ndarray | None = None,  # noqa: N803
        pw: np.ndarray | None = None,
    ) -> None:
        """Resets the (R0, p0) calibration origin."""
        rw = Rw
        if rw is None or pw is None:
            deadline = time.time() + 1.0
            rw_pw: tuple[np.ndarray, np.ndarray] | None = None
            while rw_pw is None and time.time() < deadline:
                rw_pw = tlp.read_pose(self.dev, self.source)
                if rw_pw is None:
                    time.sleep(0.002)
            if rw_pw is None:
                return
            rw, pw = rw_pw
        self._set_calibration(rw, pw)

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
        """Return diagnostic information for the active tracker transport."""
        devices = [self._device_status(name, dev) for name, dev in self._device_items()]
        selected = next((device for device in devices if device.get("selected")), None)
        source_key = self._source_key()
        return {
            "transport": self.transport,
            "endpoint": self.endpoint,
            "topic": self.topic,
            "side": self.side,
            "tracker_id": self.tracker_id,
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

    # -------------------- main read --------------------

    def read(self) -> dict[str, Any]:
        """Reads the latest pose data from the tracker."""
        if not self._has_cal and self.pose_source_available():
            self.recalibrate()

        rw_pw = tlp.read_pose(self.dev, self.source)
        if rw_pw is None:
            return {
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "event": {},
                "meta": {"id": self.name, "side": self.side, "t": time.time()},
                "valid_pose": False,
            }

        tnow = time.time()
        rw, pw = rw_pw
        rw = tlp.orthonormalize(rw)

        if not self._has_cal:
            return {
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "event": {},
                "meta": {"id": self.name, "side": self.side, "t": tnow},
                "valid_pose": False,
            }

        rloc = self.R0.T @ rw
        ploc = self.R0.T @ (pw - self.p0)

        # Axis remap: x=forward, y=left, z=up.
        axes = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
        rloc = axes @ rloc @ axes.T
        ploc = axes @ ploc

        qw, qx, qy, qz = tlp.quat_wxyz_from_R(rloc)

        return {
            "pose": [float(ploc[0]), float(ploc[1]), float(ploc[2]), qw, qx, qy, qz],
            "event": {},
            "meta": {"id": self.name, "side": self.side, "t": tnow},
            "valid_pose": True,
        }

    def get_action(self):
        """Return the latest tracker action dictionary."""
        return self.read()

    def close(self) -> None:
        """Close the underlying transport when it exposes a close method."""
        with contextlib.suppress(Exception):
            self.vr.close()


class ZMQViveTracker(BaseViveTracker):
    """Vive tracker reader backed by tracker ZMQ snapshots."""

    def __init__(
        self,
        tracker_id: str = "tracker_1",
        side: str = "center",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        haptics: bool = True,
        gripper: bool = True,
        reset_key: str = "r",
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
            tracker_id=tracker_id,
            side=side,
            source=source,
            transport="zmq",
            endpoint=endpoint,
            topic=topic,
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            haptics=haptics,
            gripper=gripper,
            reset_key=reset_key,
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

    def _resolve_tracker(self, fallback_id: str):
        dev, last_device_names = self._resolve_registry_tracker(fallback_id)
        if dev is not None:
            return dev
        raise RuntimeError(
            f"No Vive tracker samples received via ZMQ from {self.endpoint} "
            f"[{self.topic}]. Received devices: {last_device_names!r}. "
            "Is ViveTrackerZMQServer or another compatible publisher running "
            "with the same topic?",
        )


class SurviveViveTracker(BaseViveTracker):
    """Vive tracker reader backed by libsurvive/pysurvive."""

    def __init__(
        self,
        tracker_id: str = "tracker_1",
        side: str = "center",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        haptics: bool = True,
        gripper: bool = True,
        reset_key: str = "r",
    ) -> None:
        del endpoint, topic, zmq_endpoint, zmq_topic
        super().__init__(
            tracker_id=tracker_id,
            side=side,
            source=source,
            transport="survive",
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            haptics=haptics,
            gripper=gripper,
            reset_key=reset_key,
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

    def _resolve_tracker(self, fallback_id: str):
        dev, _last_device_names = self._resolve_registry_tracker(fallback_id)
        if dev is not None:
            return dev
        raise RuntimeError("No trackers found via libsurvive. Are base stations on?")


class OpenVRViveTracker(BaseViveTracker):
    """Vive tracker reader backed by SteamVR/OpenVR."""

    def __init__(
        self,
        tracker_id: str = "tracker_1",
        side: str = "center",
        *,
        source: str = "auto",
        endpoint: str | None = None,
        topic: str | None = None,
        zmq_endpoint: str | None = None,
        zmq_topic: str | None = None,
        discovery_timeout_sec: float = 3.0,
        recv_timeout_ms: int = 100,
        stale_timeout_sec: float = 0.5,
        haptics: bool = True,
        gripper: bool = True,
        reset_key: str = "r",
    ) -> None:
        del endpoint, topic, zmq_endpoint, zmq_topic
        super().__init__(
            tracker_id=tracker_id,
            side=side,
            source=source,
            transport="openvr",
            discovery_timeout_sec=discovery_timeout_sec,
            recv_timeout_ms=recv_timeout_ms,
            stale_timeout_sec=stale_timeout_sec,
            haptics=haptics,
            gripper=gripper,
            reset_key=reset_key,
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
                "Vive tracker support requires 'openvr' (pip install openvr) "
                "and a working SteamVR installation.",
            ) from _OPENVR_IMPORT_ERROR
        vr = triad_openvr()
        return vr, vr.vrsystem

    def _resolve_tracker(self, fallback_id: str):
        if openvr is None:
            raise RuntimeError(
                "Vive tracker support requires 'openvr' (pip install openvr) and a "
                "working SteamVR installation.",
            ) from _OPENVR_IMPORT_ERROR
        if fallback_id in self.vr.devices:
            return self.vr.devices[fallback_id]
        for dev in self.vr.devices.values():
            if getattr(dev, "device_class", "") in (
                "GenericTracker",
                "Tracker",
                "ViveTracker",
            ):
                return dev
        for dev in self.vr.devices.values():
            with contextlib.suppress(Exception):
                idx = getattr(dev, "index", None)
                if (
                    idx is not None
                    and self.vrsystem.getTrackedDeviceClass(idx)
                    == openvr.TrackedDeviceClass_GenericTracker
                ):
                    return dev
        raise RuntimeError("No Vive Trackers found. Is SteamVR running?")


def _tracker_class_for_transport(transport: str) -> type[BaseViveTracker]:
    normalized = BaseViveTracker._normalize_transport(transport)
    tracker_classes: dict[str, type[BaseViveTracker]] = {
        "openvr": OpenVRViveTracker,
        "survive": SurviveViveTracker,
        "zmq": ZMQViveTracker,
    }
    try:
        return tracker_classes[normalized]
    except KeyError:
        raise ValueError(f"Unsupported tracker transport: {transport!r}") from None


def ViveTracker(
    tracker_id: str = "tracker_1",
    side: str = "center",
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
    haptics: bool = True,
    gripper: bool = True,
    reset_key: str = "r",
) -> BaseViveTracker:
    """Create a Vive tracker using the requested backend transport."""
    tracker_cls = _tracker_class_for_transport(transport)
    return tracker_cls(
        tracker_id=tracker_id,
        side=side,
        source=source,
        endpoint=endpoint,
        topic=topic,
        zmq_endpoint=zmq_endpoint,
        zmq_topic=zmq_topic,
        discovery_timeout_sec=discovery_timeout_sec,
        recv_timeout_ms=recv_timeout_ms,
        stale_timeout_sec=stale_timeout_sec,
        haptics=haptics,
        gripper=gripper,
        reset_key=reset_key,
    )


class ViveTrackerZMQServer:
    """Publish raw SteamVR/OpenVR Vive tracker samples over ZMQ."""

    def __init__(
        self,
        tracker_id: str = "tracker_1",
        side: str = "center",
        *,
        bind_endpoint: str = "tcp://*:5559",
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
        self.tracker_id = tracker_id
        self.side = side.lower()
        self.bind_endpoint = bind_endpoint
        self.topic = topic
        self.rate_hz = float(rate_hz)
        self.publish_all = bool(publish_all)
        self.poll_events = bool(poll_events)
        self._selected_dev = None if self.publish_all else self._resolve_tracker()
        self._last_published_device_names: tuple[str, ...] = ()
        self._last_published_device_formats: dict[str, tuple[str, ...]] = {}

    def _is_tracker_device(self, dev) -> bool:
        if getattr(dev, "device_class", "") in (
            "GenericTracker",
            "Tracker",
            "ViveTracker",
        ):
            return True
        with contextlib.suppress(Exception):
            idx = getattr(dev, "index", None)
            return (
                idx is not None
                and self.vrsystem.getTrackedDeviceClass(idx)
                == _require_openvr().TrackedDeviceClass_GenericTracker
            )
        return False

    def _tracker_items(self) -> list[tuple[str, Any]]:
        return [
            (name, dev)
            for name, dev in sorted(self.vr.devices.items())
            if self._is_tracker_device(dev)
        ]

    def _resolve_tracker(self):
        requested = (self.tracker_id or "").strip()
        requested_lower = requested.lower()
        tracker_items = self._tracker_items()

        if requested in self.vr.devices and self._is_tracker_device(
            self.vr.devices[requested],
        ):
            return self.vr.devices[requested]

        for name, dev in tracker_items:
            serial = BaseViveTracker._device_serial(dev, name)
            if name.lower() == requested_lower or serial.lower() == requested_lower:
                return dev

        if requested_lower.startswith("tracker_"):
            suffix = requested_lower.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                idx = int(suffix) - 1
                if 0 <= idx < len(tracker_items):
                    return tracker_items[idx][1]

        for name, dev in tracker_items:
            serial = BaseViveTracker._device_serial(dev, name)
            haystacks = (name.lower(), serial.lower())
            if any(requested_lower and requested_lower in haystack for haystack in haystacks):
                return dev

        names = [name for name, _ in tracker_items]
        raise RuntimeError(
            f"No Vive tracker matching tracker_id={requested!r} found. "
            f"Discovered trackers: {names!r}.",
        )

    def _publish_name(self, name: str) -> str:
        if self.side and not self.publish_all:
            return f"tracker_{self.side}"
        return name

    def _publish_devices(self) -> dict[str, Any]:
        if self.publish_all:
            return dict(self._tracker_items())

        selected = self._selected_dev
        if selected is None:
            return {}
        for name, dev in self._tracker_items():
            if dev is selected:
                return {self._publish_name(name): dev}
        return {self._publish_name(self.tracker_id): selected}

    def publish_once(self) -> int:
        """Publish one tracker snapshot and return the number of devices sent."""
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
        """Publish tracker samples until interrupted."""
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
    "BaseViveTracker",
    "OpenVRViveTracker",
    "SurviveViveTracker",
    "ViveTracker",
    "ViveTrackerZMQServer",
    "ZMQViveTracker",
]
