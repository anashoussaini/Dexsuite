from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _import_vive_controller_module():
    try:
        return importlib.import_module("dexsuite.devices.vive_controller")
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
    return importlib.import_module("dexsuite.devices.vive_controller")


vive_controller = _import_vive_controller_module()


class FakeDevice:
    def __init__(self, pose, buttons: dict | None = None):
        self.pose = pose
        self.buttons = {} if buttons is None else dict(buttons)
        self.haptic_calls = 0

    def get_pose_quaternion(self):
        return None if self.pose is None else list(self.pose)

    def get_serial(self):
        return "fake-controller-serial"

    def get_controller_inputs(self):
        return dict(self.buttons)

    def trigger_haptic_pulse(self, duration_micros: int = 1000, axis_id: int = 0):
        del duration_micros, axis_id
        self.haptic_calls += 1


class FakeController(vive_controller.BaseViveController):
    def __init__(
        self,
        device: FakeDevice,
        *,
        callback=None,
        haptics: bool = False,
    ):
        self._fake_device = device
        super().__init__(
            controller_id="controller_1",
            side="left",
            transport="fake",
            discovery_timeout_sec=0.0,
            gripper_callback=callback,
            haptics=haptics,
        )

    def _init_backend(self, **kwargs):
        del kwargs
        return SimpleNamespace(devices={"controller_1": self._fake_device}), None

    def _resolve_controller(self, side: str, fallback_id: str):
        del side, fallback_id
        return self._fake_device


def test_base_recalibration_stores_initial_pose():
    device = FakeDevice([0.25, -0.5, 1.25, 1.0, 0.0, 0.0, 0.0])

    controller = FakeController(device)

    assert controller.calibrated is True
    assert np.allclose(controller.p0, [0.25, -0.5, 1.25])
    assert np.allclose(controller.R0, np.eye(3))
    assert device.haptic_calls == 0


def test_base_read_returns_zero_motion_after_calibration():
    device = FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    controller = FakeController(device)

    action = controller.read()

    assert action["pose"] == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert action["buttons"] == {}
    assert action["gripper"] == [0]


def test_base_read_converts_changed_pose_to_local_action():
    device = FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    controller = FakeController(device)
    device.pose = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    action = controller.read()

    assert len(action["pose"]) == 7
    assert action["pose"][:3] == pytest.approx([1.0, 0.0, 0.0])
    assert action["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_missing_pose_returns_neutral_pose_and_preserves_buttons():
    device = FakeDevice(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        {"trigger": 1.0},
    )
    controller = FakeController(device)
    device.pose = None

    action = controller.read()

    assert action["pose"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert action["buttons"] == {"trigger": 1.0}
    assert action["gripper"] == [1]


def test_button_parsing_invokes_gripper_callback_on_trigger_zone_change():
    callback_actions = []
    device = FakeDevice(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        {"trigger": 1.0},
    )
    controller = FakeController(device, callback=callback_actions.append)

    action = controller.read()

    assert action["gripper"] == [1]
    assert callback_actions == [1]


@pytest.mark.parametrize(
    ("transport", "class_name"),
    [
        ("openvr", "OpenVRViveController"),
        ("survive", "SurviveViveController"),
        ("zmq", "ZMQViveController"),
        ("remote", "ZMQViveController"),
        ("bridge", "ZMQViveController"),
        ("zmq_client", "ZMQViveController"),
    ],
)
def test_vive_controller_factory_selects_backend(monkeypatch, transport, class_name):
    class DummyController:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(vive_controller, class_name, DummyController)

    controller = vive_controller.ViveController(
        "controller_custom",
        side="right",
        transport=transport,
        endpoint="tcp://example:5558",
        topic="custom.topic",
    )

    assert isinstance(controller, DummyController)
    assert controller.args == ()
    assert controller.kwargs["controller_id"] == "controller_custom"
    assert controller.kwargs["side"] == "right"
    assert controller.kwargs["endpoint"] == "tcp://example:5558"
    assert controller.kwargs["topic"] == "custom.topic"


def test_vive_controller_factory_rejects_unknown_transport():
    with pytest.raises(ValueError, match="Unsupported controller transport"):
        vive_controller.ViveController(transport="unknown")


def test_zmq_vive_controller_accepts_explicit_bridge_settings(monkeypatch):
    captured = {}

    def fake_init_backend(self, **kwargs):
        del self
        captured["kwargs"] = kwargs
        return SimpleNamespace(devices={}), None

    def fake_resolve(self, side, fallback_id):
        del self, side, fallback_id
        return FakeDevice([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(vive_controller.ZMQViveController, "_init_backend", fake_init_backend)
    monkeypatch.setattr(vive_controller.ZMQViveController, "_resolve_controller", fake_resolve)

    controller = vive_controller.ZMQViveController(
        endpoint=vive_controller.DEFAULT_BRIDGE_ENDPOINT,
        topic=vive_controller.DEFAULT_BRIDGE_TOPIC,
    )

    assert controller.endpoint == vive_controller.DEFAULT_BRIDGE_ENDPOINT
    assert controller.topic == vive_controller.DEFAULT_BRIDGE_TOPIC
    assert captured["kwargs"]["endpoint"] == vive_controller.DEFAULT_BRIDGE_ENDPOINT
    assert captured["kwargs"]["topic"] == vive_controller.DEFAULT_BRIDGE_TOPIC


def test_package_exports_only_new_zmq_controller():
    from dexsuite.devices import (  # noqa: PLC0415
        BaseViveController,
        OpenVRViveController,
        SurviveViveController,
        ZMQViveController,
    )

    assert BaseViveController is vive_controller.BaseViveController
    assert OpenVRViveController is vive_controller.OpenVRViveController
    assert SurviveViveController is vive_controller.SurviveViveController
    assert ZMQViveController is vive_controller.ZMQViveController
    assert not hasattr(vive_controller, "ViveControllerZMQClient")
    assert not hasattr(vive_controller, "RemoteViveController")


def test_vive_controller_zmq_module_only_exports_new_client():
    module = importlib.import_module("dexsuite.devices.vive_controller_zmq")

    assert module.ZMQViveController is vive_controller.ZMQViveController
    assert module.ViveControllerZMQServer is vive_controller.ViveControllerZMQServer
    assert not hasattr(module, "ViveControllerZMQClient")
    assert not hasattr(module, "RemoteViveController")
