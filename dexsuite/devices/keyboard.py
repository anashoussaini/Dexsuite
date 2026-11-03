"""Keyboard device helper for real-time action input.

This version:
- Works on Linux/Windows/macOS with pynput.
- Avoids AttributeError on release by safely extracting chars.
- Normalizes chars to lowercase.
- Provides QWERTY and AZERTY mapping presets.
"""

from pynput.keyboard import Key, KeyCode, Listener

from dexsuite.devices.device import Device

MAPPING_AZERTY = {
    "pos": [("z", "s"), ("q", "d"), ("e", "a")],
    "rot": [("h", "k"), ("j", "u"), ("y", "i")],
    "gripper": [("p", "o")],
    "reset": [("r", "r")],
}

MAPPING_QWERTY = {
    "pos": [(Key.up, Key.down), (Key.left, Key.right), ("u", "j")],
    "rot": [("b", "m"), ("n", "h"), ("g", "v")],
    "gripper": [("p", "o")],
    "reset": [("r", "r")],
}

DEFAULT_MAPPING = MAPPING_QWERTY  # keep your default; pass MAPPING_AZERTY to switch


class Keyboard(Device):
    """Listen to keyboard input and produce action dictionaries."""

    def __init__(
        self,
        controller: str,
        normalized: bool,
        pos_sens: float = 1.0,
        rot_sens: float = 1.0,
        mapping: dict = DEFAULT_MAPPING,
        gripper: bool = False,
    ):
        super().__init__(controller, normalized, gripper)
        if self.controller != "OSCPose":
            raise ValueError("Wrong controller entered. Keyboard only accepts OSCPose")

        self.mapping = dict(mapping)  # copy to be safe
        self.action = {"pose": [0, 0, 0, 0, 0, 0], "event": None, "meta": {}}
        if gripper:
            self.action["gripper"] = [0]
        if mapping.get("reset") is not None:
            self.action["reset"] = [0]

        # sensitivities
        if normalized:
            self.pos_sens = min(pos_sens, 1.0)
            self.rot_sens = min(rot_sens, 1.0)
        else:
            self.pos_sens = pos_sens
            self.rot_sens = rot_sens

        # listener
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    @staticmethod
    def _get_char(key):
        """Return the lowercase character for Key/KeyCode or None for special keys."""
        if isinstance(key, KeyCode) and key.char:
            return key.char.lower()
        elif isinstance(key, Key):  # handle special keys like arrows
            return key
        return None

    def on_press(self, key):
        char = self._get_char(key)
        if not char:
            return
        self.action["event"] = str(char)

        for name, mapping in self.mapping.items():
            sens = 1.0
            offset = 0
            if name == "pos":
                sens = self.pos_sens
                name = "pose"  # map to action key
            elif name == "rot":
                sens = self.rot_sens
                name = "pose"
                offset = 3

            for i, keys in enumerate(mapping):
                if char == keys[0]:
                    self.action[name][i + offset] = sens
                elif char == keys[1]:
                    self.action[name][i + offset] = -sens

    def on_release(self, key):
        char = self._get_char(key)
        if char is None:
            return
        self.action["event"] = None

        for name, mapping in self.mapping.items():
            offset = 0
            if name == "gripper":
                for i, keys in enumerate(mapping):
                    if char in keys:
                        self.action[name][i] = 0
                continue
            if name == "rot":
                name = "pose"
                offset = 3
            elif name == "pos":
                name = "pose"

            for i, keys in enumerate(mapping):
                if char in keys:
                    self.action[name][i + offset] = 0
