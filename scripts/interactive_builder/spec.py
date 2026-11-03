"""Serializable specification for building and running DexSuite environments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

POSE_BACKEND_CONTROLLERS = frozenset(
    {"osc_pose", "osc_pose_abs", "osc_pose_abs_quat", "ik_pose"},
)
POSE_BACKENDS = ("mink", "genesis_internal")


def controller_supports_backend(controller: str | None) -> bool:
    """Return whether a controller exposes selectable pose backends."""
    return str(controller or "") in POSE_BACKEND_CONTROLLERS


def arm_backend(arm: ArmSpec) -> str:
    """Return the selected backend for an arm pose controller."""
    if not controller_supports_backend(arm.arm_control):
        return ""
    backend = str(arm.arm_control_config.get("backend", POSE_BACKENDS[0]))
    return backend if backend in POSE_BACKENDS else POSE_BACKENDS[0]


def set_arm_backend(arm: ArmSpec, backend: str) -> None:
    """Set or clear backend config for an arm controller."""
    if not controller_supports_backend(arm.arm_control):
        arm.arm_control_config.pop("backend", None)
        return
    backend = str(backend)
    if backend not in POSE_BACKENDS:
        backend = POSE_BACKENDS[0]
    arm.arm_control_config["backend"] = backend


@dataclass(slots=True)
class ArmSpec:
    """High-level configuration for one arm (single, left, or right)."""

    manipulator: str
    gripper: str | None
    arm_control: str
    gripper_control: str | None
    arm_control_config: dict[str, Any] = field(default_factory=dict)
    gripper_control_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the arm spec to a JSON-compatible dictionary."""
        return {
            "manipulator": self.manipulator,
            "gripper": self.gripper,
            "arm_control": self.arm_control,
            "gripper_control": self.gripper_control,
            "arm_control_config": dict(self.arm_control_config),
            "gripper_control_config": dict(self.gripper_control_config),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmSpec:
        """Build an arm spec from a JSON-compatible dictionary."""
        return cls(
            manipulator=str(d["manipulator"]),
            gripper=None if d.get("gripper") is None else str(d["gripper"]),
            arm_control=str(d["arm_control"]),
            gripper_control=(
                None if d.get("gripper_control") is None else str(d["gripper_control"])
            ),
            arm_control_config=dict(d.get("arm_control_config") or {}),
            gripper_control_config=dict(d.get("gripper_control_config") or {}),
        )


RobotType = Literal["single", "bimanual"]


@dataclass(slots=True)
class BuilderSpec:
    """Top-level, JSON-friendly spec for dexsuite.make.

    Conventions:
        - cameras is None means use defaults (front + wrist).
        - cameras is [] means disable cameras (pass cameras=None to the API).
        - cameras is a non-empty list means use exactly those camera names.
    """

    task: str = "reach"
    type_of_robot: RobotType = "single"

    single: ArmSpec | None = None
    left: ArmSpec | None = None
    right: ArmSpec | None = None
    layout_preset: str | None = None

    control_hz: int = 20
    n_envs: int = 1
    performance_mode: bool = False

    cameras: list[str] | None = None
    modalities: tuple[str, ...] = ("rgb",)
    render_mode: Literal["human", "rgb_array"] | None = "human"
    input_device: str = "keyboard"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the builder spec to a JSON-compatible dictionary."""
        return {
            "task": self.task,
            "type_of_robot": self.type_of_robot,
            "single": None if self.single is None else self.single.to_dict(),
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
            "layout_preset": self.layout_preset,
            "control_hz": int(self.control_hz),
            "n_envs": int(self.n_envs),
            "performance_mode": bool(self.performance_mode),
            "cameras": self.cameras,
            "modalities": list(self.modalities),
            "render_mode": self.render_mode,
            "input_device": self.input_device,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuilderSpec:
        """Build a builder spec from a JSON-compatible dictionary."""
        return cls(
            task=str(d["task"]),
            type_of_robot=str(d["type_of_robot"]).lower(),  # type: ignore[arg-type]
            single=None if d.get("single") is None else ArmSpec.from_dict(d["single"]),
            left=None if d.get("left") is None else ArmSpec.from_dict(d["left"]),
            right=None if d.get("right") is None else ArmSpec.from_dict(d["right"]),
            layout_preset=None
            if d.get("layout_preset") is None
            else str(d["layout_preset"]),
            control_hz=int(d.get("control_hz", 20)),
            n_envs=int(d.get("n_envs", 1)),
            performance_mode=bool(d.get("performance_mode", False)),
            cameras=(None if d.get("cameras") is None else list(d.get("cameras"))),
            modalities=tuple(d.get("modalities") or ("rgb",)),
            render_mode=d.get("render_mode"),
            input_device=str(d.get("input_device", "keyboard")),
        )

    def to_json(self, path: Path) -> None:
        """Write this builder spec to a formatted JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> BuilderSpec:
        """Read a builder spec from a JSON file."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # --------------------------- DexSuite integration ---------------------------
    def _to_arm_options(self, arm: ArmSpec):
        from dexsuite.options import ArmOptions, ControllerOptions

        arm_ctrl = ControllerOptions(
            name=arm.arm_control,
            config=dict(arm.arm_control_config),
        )
        grip_ctrl = (
            None
            if arm.gripper_control is None
            else ControllerOptions(
                name=arm.gripper_control,
                config=dict(arm.gripper_control_config),
            )
        )
        return ArmOptions(
            manipulator=arm.manipulator,
            gripper=arm.gripper,
            manipulator_controller=arm_ctrl,
            gripper_controller=grip_ctrl,
        )

    def to_make_kwargs(self) -> dict[str, Any]:
        """Convert this spec to keyword arguments for dexsuite.make."""
        from dexsuite.options import LayoutOptions, RobotOptions, SimOptions

        sim = SimOptions(
            control_hz=int(self.control_hz),
            performance_mode=bool(self.performance_mode),
            n_envs=int(self.n_envs),
        )

        if self.type_of_robot == "single":
            if self.single is None:
                raise ValueError(
                    "BuilderSpec.single must be set for type_of_robot='single'.",
                )
            robot = RobotOptions(
                type_of_robot="single",
                single=self._to_arm_options(self.single),
            )
        elif self.type_of_robot == "bimanual":
            if self.left is None or self.right is None:
                raise ValueError(
                    "BuilderSpec.left/right must be set for type_of_robot='bimanual'.",
                )
            layout = (
                LayoutOptions(preset=self.layout_preset)
                if self.layout_preset
                else LayoutOptions()
            )
            robot = RobotOptions(
                type_of_robot="bimanual",
                left=self._to_arm_options(self.left),
                right=self._to_arm_options(self.right),
                layout=layout,
            )
        else:
            raise ValueError(f"Unknown type_of_robot: {self.type_of_robot!r}")

        kw: dict[str, Any] = {
            "sim": sim,
            "robot": robot,
            "render_mode": self.render_mode,
            "modalities": self.modalities,
        }

        # Cameras: None uses defaults, [] disables, list selects explicitly.
        if self.cameras is None:
            return kw
        if len(self.cameras) == 0:
            kw["cameras"] = None
            return kw
        kw["cameras"] = list(self.cameras)
        return kw
