from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _import_vive_tracker_module():
    try:
        return importlib.import_module("dexsuite.devices.vive_tracker")
    except ModuleNotFoundError as exc:
        if exc.name not in {"genesis", "torch"}:
            raise

    for name in list(sys.modules):
        if name == "dexsuite" or name.startswith("dexsuite."):
            sys.modules.pop(name, None)

    repo_root = Path(__file__).resolve().parents[2]
    pkg = ModuleType("dexsuite")
    pkg.__path__ = [str(repo_root / "dexsuite")]
    sys.modules["dexsuite"] = pkg

    devices_path = repo_root / "dexsuite" / "devices"
    spec = importlib.util.spec_from_file_location(
        "dexsuite.devices",
        devices_path / "__init__.py",
        submodule_search_locations=[str(devices_path)],
    )
    assert spec is not None and spec.loader is not None
    devices_pkg = importlib.util.module_from_spec(spec)
    sys.modules["dexsuite.devices"] = devices_pkg
    spec.loader.exec_module(devices_pkg)
    return importlib.import_module("dexsuite.devices.vive_tracker")


vive_tracker = _import_vive_tracker_module()


class FakeDevice:
    def __init__(self, pose):
        self.pose = pose
        self.haptic_calls = 0

    def get_pose_quaternion(self):
        return None if self.pose is None else list(self.pose)

    def get_serial(self):
        return "fake-serial"

    def trigger_haptic_pulse(self, duration_micros: int = 1000, axis_id: int = 0):
        del duration_micros, axis_id
        self.haptic_calls += 1


class FakeTracker(vive_tracker.BaseViveTracker):
    def __init__(self, device: FakeDevice):
        self._fake_device = device
        super().__init__(
            tracker_id="tracker_1",
            side="center",
            transport="fake",
            discovery_timeout_sec=0.0,
            haptics=False,
        )

    def _init_backend(self, **kwargs):
        del kwargs
        return SimpleNamespace(devices={"tracker_1": self._fake_device}), None

    def _resolve_tracker(self, fallback_id: str):
        del fallback_id
        return self._fake_device


def test_base_recalibration_stores_initial_pose():
    device = FakeDevice([0.25, -0.5, 1.25, 1.0, 0.0, 0.0, 0.0])

    tracker = FakeTracker(device)

    assert tracker.calibrated is True
    assert np.allclose(tracker.p0, [0.25, -0.5, 1.25])
    assert np.allclose(tracker.R0, np.eye(3))
    assert device.haptic_calls == 0


def test_base_read_returns_zero_motion_after_calibration():
    device = FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tracker = FakeTracker(device)

    action = tracker.read()

    assert action["valid_pose"] is True
    assert action["pose"] == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_base_read_converts_changed_pose_to_local_action():
    device = FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tracker = FakeTracker(device)
    device.pose = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    action = tracker.read()

    assert action["valid_pose"] is True
    assert len(action["pose"]) == 7
    assert action["pose"][:3] == pytest.approx([1.0, 0.0, 0.0])
    assert action["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_base_read_returns_neutral_action_when_pose_is_missing():
    device = FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tracker = FakeTracker(device)
    device.pose = None

    action = tracker.read()

    assert action["valid_pose"] is False
    assert action["pose"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("transport", "class_name"),
    [
        ("openvr", "OpenVRViveTracker"),
        ("survive", "SurviveViveTracker"),
        ("zmq", "ZMQViveTracker"),
        ("remote", "ZMQViveTracker"),
        ("bridge", "ZMQViveTracker"),
        ("zmq_client", "ZMQViveTracker"),
    ],
)
def test_vive_tracker_factory_selects_backend(monkeypatch, transport, class_name):
    class DummyTracker:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(vive_tracker, class_name, DummyTracker)

    tracker = vive_tracker.ViveTracker(
        "tracker_custom",
        side="left",
        transport=transport,
        endpoint="tcp://example:5558",
        topic="custom.topic",
    )

    assert isinstance(tracker, DummyTracker)
    assert tracker.args == ()
    assert tracker.kwargs["tracker_id"] == "tracker_custom"
    assert tracker.kwargs["side"] == "left"
    assert tracker.kwargs["endpoint"] == "tcp://example:5558"
    assert tracker.kwargs["topic"] == "custom.topic"


def test_vive_tracker_factory_rejects_unknown_transport():
    with pytest.raises(ValueError, match="Unsupported tracker transport"):
        vive_tracker.ViveTracker(transport="unknown")


def test_zmq_vive_tracker_accepts_explicit_bridge_settings(monkeypatch):
    captured = {}

    def fake_init_backend(self, **kwargs):
        del self
        captured["kwargs"] = kwargs
        return SimpleNamespace(devices={}), None

    def fake_resolve(self, fallback_id):
        del self, fallback_id
        return FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(vive_tracker.ZMQViveTracker, "_init_backend", fake_init_backend)
    monkeypatch.setattr(vive_tracker.ZMQViveTracker, "_resolve_tracker", fake_resolve)

    tracker = vive_tracker.ZMQViveTracker(
        endpoint=vive_tracker.DEFAULT_BRIDGE_ENDPOINT,
        topic=vive_tracker.DEFAULT_BRIDGE_TOPIC,
    )

    assert tracker.endpoint == vive_tracker.DEFAULT_BRIDGE_ENDPOINT
    assert tracker.topic == vive_tracker.DEFAULT_BRIDGE_TOPIC
    assert captured["kwargs"]["endpoint"] == vive_tracker.DEFAULT_BRIDGE_ENDPOINT
    assert captured["kwargs"]["topic"] == vive_tracker.DEFAULT_BRIDGE_TOPIC


def test_package_exports_only_new_zmq_tracker():
    from dexsuite.devices import (  # noqa: PLC0415
        BaseViveTracker,
        OpenVRViveTracker,
        SurviveViveTracker,
        ZMQViveTracker,
    )

    assert BaseViveTracker is vive_tracker.BaseViveTracker
    assert OpenVRViveTracker is vive_tracker.OpenVRViveTracker
    assert SurviveViveTracker is vive_tracker.SurviveViveTracker
    assert ZMQViveTracker is vive_tracker.ZMQViveTracker
    assert not hasattr(vive_tracker, "ViveTrackerZMQClient")
    assert not hasattr(vive_tracker, "RemoteViveTracker")


def test_vive_tracker_zmq_module_only_exports_new_client():
    module = importlib.import_module("dexsuite.devices.vive_tracker_zmq")

    assert module.ZMQViveTracker is vive_tracker.ZMQViveTracker
    assert module.ViveTrackerZMQServer is vive_tracker.ViveTrackerZMQServer
    assert not hasattr(module, "ViveTrackerZMQClient")
    assert not hasattr(module, "RemoteViveTracker")
