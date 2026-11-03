"""3Dconnexion SpaceMouse reader with cross-platform backends.

Backend selection:
- Linux: evdev (reads /dev/input/eventX)
- macOS / Windows (and Linux fallback): hid (HIDAPI wrapper)
"""

from __future__ import annotations

import contextlib
import struct
import sys
import threading
import time
from typing import Any

from dexsuite.devices.device import Device

VENDOR_3DCONNEXION = 0x256F  # common 3Dconnexion vendor ID


class Spacemouse(Device):
    """Read 6-DoF SpaceMouse input and expose it as a DexSuite action dict."""

    def __init__(
        self,
        path_or_ids: object,
        controller: str,
        normalized: bool,
        pos_sens: float = 1.0,
        rot_sens: float = 1.0,
        gripper: bool = True,
        deadzone: float = 0.0,
    ) -> None:
        """Create a SpaceMouse reader.

        Args:
            path_or_ids: Backend selector / device locator.
                - Linux evdev: "/dev/input/eventX"
                - HID backend: a bytes path returned by hid.enumerate()
                - Tuple[int, int]: (vendor_id, product_id)
                - None/other: auto-select the first matching device.
            controller: Must be "OSCPose" (6D pose delta).
            normalized: If True, Device.get_action() clips pose to [-1, 1].
            pos_sens: Translation sensitivity multiplier.
            rot_sens: Rotation sensitivity multiplier.
            gripper: If True, exposes "gripper" and maps a button toggle to it.
            deadzone: Deadzone threshold.
                - If > 1.0, interpreted as raw counts (HID) or raw abs units (evdev).
                - If <= 1.0, interpreted as normalized units in [-1, 1].
        """
        super().__init__(controller, normalized, gripper)
        if controller != "OSCPose":
            raise ValueError("Spacemouse only supports controller='OSCPose'.")

        self.pos_sens = float(pos_sens)
        self.rot_sens = float(rot_sens)
        self.deadzone = float(deadzone)

        self._backend = self._select_backend(path_or_ids)
        self._evdev: dict[str, Any] | None = None
        self._hid: Any | None = None
        self._hid_buttons_prev: int = 0

        self.dev: Any | None = None
        self.run = False
        self.t: threading.Thread | None = None

        self.gripper_state = False
        self.action = {
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reset": [0],
            "event": None,
            "meta": {"path": path_or_ids, "backend": self._backend},
        }
        if self.gripper:
            self.action["gripper"] = [0.0]

        self._axis_values = [0.0] * 6  # normalized [-1, 1] for both backends
        self._axis_thres = [0.0] * 6  # per-axis normalized threshold

        # evdev-only calibration
        self._axis_codes: list[int] = []
        self._code_to_index: dict[int, int] = {}
        self._axis_scale = [1.0] * 6  # 1/max_range per axis
        self._axis_zero = [0.0] * 6  # baseline centers (raw)

        self._open_device(path_or_ids)
        self._init_axis_calibration()

        self.run = True
        self.t = threading.Thread(target=self._read_loop, daemon=True)
        self.t.start()

    # ------------------------- backend selection -------------------------

    def _select_backend(self, path_or_ids: object) -> str:
        # Explicit evdev path forces evdev backend.
        if isinstance(path_or_ids, (bytes, bytearray, str)):
            path = (
                path_or_ids.decode("utf-8", errors="ignore")
                if isinstance(path_or_ids, (bytes, bytearray))
                else str(path_or_ids)
            )
            if "/dev/input/event" in path:
                return "evdev"

        if sys.platform.startswith("linux") and self._can_import("evdev"):
            return "evdev"

        if self._can_import("hid"):
            return "hid"

        if sys.platform.startswith("linux"):
            raise RuntimeError(
                "SpaceMouse support requires 'evdev' (Linux) or a HIDAPI wrapper "
                "providing the 'hid' module (pip install hidapi).",
            )
        raise RuntimeError(
            "SpaceMouse support requires a HIDAPI wrapper providing the 'hid' module "
            "(pip install hidapi).",
        )

    @staticmethod
    def _can_import(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    # ------------------------- device open / find -------------------------

    def _open_device(self, path_or_ids: object) -> None:
        if self._backend == "evdev":
            self._open_evdev(path_or_ids)
        else:
            self._open_hid(path_or_ids)

    def _open_evdev(self, path_or_ids: object) -> None:
        try:
            from evdev import InputDevice, ecodes, list_devices  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SpaceMouse evdev backend requires 'evdev' (pip install evdev).",
            ) from e

        self._evdev = {
            "InputDevice": InputDevice,
            "ecodes": ecodes,
            "list_devices": list_devices,
        }

        # Axis mapping/order from evdev
        self._axis_codes = [
            ecodes.ABS_X,  # index 0
            ecodes.ABS_Y,  # index 1
            ecodes.ABS_Z,  # index 2
            ecodes.ABS_RX,  # index 3
            ecodes.ABS_RY,  # index 4
            ecodes.ABS_RZ,  # index 5
        ]
        self._code_to_index = {c: i for i, c in enumerate(self._axis_codes)}

        dev = None
        if isinstance(path_or_ids, (bytes, bytearray, str)):
            path = (
                path_or_ids.decode("utf-8", errors="ignore")
                if isinstance(path_or_ids, (bytes, bytearray))
                else str(path_or_ids)
            )
            if "/event" in path:
                dev = InputDevice(path)
            else:
                dev = self._find_evdev_candidate()
        elif (
            isinstance(path_or_ids, tuple)
            and len(path_or_ids) == 2
            and all(isinstance(x, int) for x in path_or_ids)
        ):
            vid, pid = path_or_ids
            dev = self._find_evdev_candidate(vid=int(vid), pid=int(pid))
        else:
            dev = self._find_evdev_candidate()

        if dev is None:
            raise OSError("No suitable 3Dconnexion SpaceMouse evdev interface found.")

        # Optional: grab for exclusive access (ignore errors)
        with contextlib.suppress(Exception):
            if hasattr(dev, "grab"):
                dev.grab()

        # Nonblocking read
        if hasattr(dev, "set_nonblocking"):
            dev.set_nonblocking(True)

        self.dev = dev

    def _find_evdev_candidate(
        self, *, vid: int = VENDOR_3DCONNEXION, pid: int | None = None,
    ):
        assert self._evdev is not None
        InputDevice = self._evdev["InputDevice"]
        ecodes = self._evdev["ecodes"]
        list_devices = self._evdev["list_devices"]

        best = None
        for path in list_devices():
            with contextlib.suppress(Exception):
                dev = InputDevice(path)
                name = (getattr(dev, "name", "") or "").lower()
                info = getattr(dev, "info", None)
                v_id = getattr(info, "vendor", None)
                p_id = getattr(info, "product", None)

                if vid is not None and v_id != vid:
                    continue
                if pid is not None and p_id != pid:
                    continue
                if "keyboard" in name or "kmj" in name:
                    continue
                caps = dev.capabilities(absinfo=False)
                if ecodes.EV_ABS not in caps:
                    continue

                best = dev
                if "3dconnexion" in name or "spacemouse" in name:
                    return dev
        return best

    def _open_hid(self, path_or_ids: object) -> None:
        try:
            import hid  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SpaceMouse HID backend requires a HIDAPI wrapper providing the 'hid' module "
                "(pip install hidapi).",
            ) from e

        self._hid = hid

        path = None
        vid = VENDOR_3DCONNEXION
        pid = None

        if isinstance(path_or_ids, (bytes, bytearray)):
            path = bytes(path_or_ids)
        elif isinstance(path_or_ids, str):
            # Some platforms expose the HID path as bytes; accept str as a best-effort.
            path = path_or_ids.encode("utf-8", errors="ignore")
        elif (
            isinstance(path_or_ids, tuple)
            and len(path_or_ids) == 2
            and all(isinstance(x, int) for x in path_or_ids)
        ):
            vid, pid = int(path_or_ids[0]), int(path_or_ids[1])

        if path is None:
            path = self._find_hid_candidate_path(vid=vid, pid=pid)

        if path is None:
            raise OSError("No suitable 3Dconnexion SpaceMouse HID interface found.")

        dev = hid.device()
        dev.open_path(path)
        with contextlib.suppress(Exception):
            dev.set_nonblocking(1)
        self.dev = dev

    def _find_hid_candidate_path(self, *, vid: int, pid: int | None) -> bytes | None:
        assert self._hid is not None
        items: list[dict[str, Any]] | None

        # hid.enumerate signature varies by wrapper; try a few patterns.
        items = None
        try:
            items = self._hid.enumerate(vid, pid or 0)  # type: ignore[arg-type]
        except Exception:
            items = None
        if not items:
            try:
                items = self._hid.enumerate(vid)  # type: ignore[arg-type]
            except Exception:
                items = None
        if not items:
            try:
                items = self._hid.enumerate()  # type: ignore[call-arg]
            except Exception:
                items = None
            if items:
                items = [d for d in items if int(d.get("vendor_id", 0)) == vid]

        if not items:
            return None

        best = None
        for d in items:
            if pid is not None and int(d.get("product_id", 0)) != int(pid):
                continue
            name = d.get("product_string") or d.get("manufacturer_string") or ""
            name = str(name).lower()
            if "3dconnexion" in name or "spacemouse" in name:
                best = d
                break
            best = d
        if not best:
            return None
        return best.get("path")

    # ------------------------------ calibration -----------------------------

    def _init_axis_calibration(self) -> None:
        if self._backend == "evdev":
            self._init_evdev_calibration()
        else:
            self._init_hid_calibration()

    def _init_evdev_calibration(self) -> None:
        assert self._evdev is not None
        for i, code in enumerate(self._axis_codes):
            try:
                ai = self.dev.absinfo(code)
            except Exception:
                ai = None

            if ai is not None:
                rng = max(abs(getattr(ai, "min", -512)), abs(getattr(ai, "max", 512)))
                rng = float(rng if rng else 512.0)
                self._axis_scale[i] = 1.0 / rng
                self._axis_zero[i] = float(getattr(ai, "value", 0))
                self._axis_thres[i] = (
                    min(1.0, self.deadzone / rng)
                    if self.deadzone > 1.0
                    else self.deadzone
                )
            else:
                self._axis_scale[i] = 1.0 / 512.0
                self._axis_zero[i] = 0.0
                self._axis_thres[i] = (
                    self.deadzone if self.deadzone <= 1.0 else self.deadzone / 512.0
                )

    def _init_hid_calibration(self) -> None:
        # Typical SpaceMouse range is roughly +/-350 counts. Keep conservative defaults.
        scale = 1.0 / 350.0
        thres = self.deadzone * scale if self.deadzone > 1.0 else self.deadzone
        self._axis_thres = [float(thres)] * 6
        self._axis_scale = [scale] * 6
        self._axis_zero = [0.0] * 6

    # ------------------------------ lifecycle -----------------------------

    def close(self) -> None:
        self.run = False
        if self.t and self.t.is_alive():
            for _ in range(10):
                if not self.t.is_alive():
                    break
                time.sleep(0.02)

        if self._backend == "evdev":
            with contextlib.suppress(Exception):
                if self.dev and hasattr(self.dev, "ungrab"):
                    self.dev.ungrab()
        else:
            with contextlib.suppress(Exception):
                if self.dev and hasattr(self.dev, "close"):
                    self.dev.close()

        self.dev = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # ---------------------------- event reading ---------------------------

    def _read_loop(self) -> None:
        if self._backend == "evdev":
            self._read_evdev()
        else:
            self._read_hid()

    def _read_evdev(self) -> None:
        assert self._evdev is not None
        ecodes = self._evdev["ecodes"]

        btn_left = getattr(ecodes, "BTN_0", 0x100)
        btn_right = getattr(ecodes, "BTN_1", 0x101)

        while self.run:
            if not self.dev:
                time.sleep(0.005)
                continue

            try:
                ev = self.dev.read_one()
            except BlockingIOError:
                ev = None
            except OSError:
                break
            except Exception:
                ev = None

            if ev is None:
                time.sleep(0.002)
                continue

            if ev.type == ecodes.EV_ABS and ev.code in self._code_to_index:
                idx = self._code_to_index[ev.code]
                norm = (float(ev.value) - self._axis_zero[idx]) * self._axis_scale[idx]
                if abs(norm) < self._axis_thres[idx]:
                    norm = 0.0
                self._axis_values[idx] = max(-1.0, min(1.0, norm))
                self._update_action_from_axes()
            elif ev.type == ecodes.EV_KEY:
                if ev.code == btn_right and ev.value == 1 and self.gripper:
                    self._toggle_gripper()
                elif ev.code == btn_left:
                    self.action["reset"] = [1 if ev.value == 1 else 0]

    def _read_hid(self) -> None:
        # HID report IDs used by many 3Dconnexion devices:
        # 1: translation, 2: rotation, 3: buttons
        left_mask = 1 << 0
        right_mask = 1 << 1

        while self.run:
            if not self.dev:
                time.sleep(0.005)
                continue

            try:
                data = self.dev.read(64)
            except Exception:
                data = None

            if not data:
                time.sleep(0.002)
                continue

            buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
            if not buf:
                time.sleep(0.002)
                continue

            report_id = buf[0]
            if report_id in (1, 2) and len(buf) >= 7:
                x, y, z = struct.unpack_from("<hhh", buf, 1)
                if report_id == 1:
                    self._set_axis(0, x)
                    self._set_axis(1, y)
                    self._set_axis(2, z)
                else:
                    self._set_axis(3, x)
                    self._set_axis(4, y)
                    self._set_axis(5, z)
                self._update_action_from_axes()
            elif report_id == 3:
                buttons = 0
                if len(buf) >= 3:
                    buttons = int.from_bytes(buf[1:3], "little")
                elif len(buf) >= 2:
                    buttons = int(buf[1])

                left = bool(buttons & left_mask)
                right = bool(buttons & right_mask)

                if self.gripper and right and not (self._hid_buttons_prev & right_mask):
                    self._toggle_gripper()

                self.action["reset"] = [1 if left else 0]
                self._hid_buttons_prev = buttons

    def _set_axis(self, idx: int, raw: int) -> None:
        norm = (float(raw) - self._axis_zero[idx]) * self._axis_scale[idx]
        if abs(norm) < self._axis_thres[idx]:
            norm = 0.0
        self._axis_values[idx] = max(-1.0, min(1.0, norm))

    def _toggle_gripper(self) -> None:
        self.gripper_state = not self.gripper_state
        self.action["gripper"] = [1.0 if self.gripper_state else -1.0]

    def _update_action_from_axes(self) -> None:
        # Translation: [-ty, -tx, -tz]
        tx, ty, tz = self._axis_values[0], self._axis_values[1], self._axis_values[2]
        pos = [-ty * self.pos_sens, -tx * self.pos_sens, -tz * self.pos_sens]

        # Rotation: roll, pitch, yaw = [-ry, -rx, -rz]
        rx, ry, rz = self._axis_values[3], self._axis_values[4], self._axis_values[5]
        rot = [(-ry) * self.rot_sens, (-rx) * self.rot_sens, (-rz) * self.rot_sens]

        self.action["pose"][:3] = pos
        self.action["pose"][3:] = rot
