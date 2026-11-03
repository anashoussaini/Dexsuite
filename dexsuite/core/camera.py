"""Integrate Genesis cameras with DexSuite observations.

Camera positions use world-frame meters, quaternions use WXYZ order, and rendered tensors are batched over environments. Dynamic cameras attach to gripper links after scene build and can expose RGB, depth, segmentation, and normal modalities.
"""

from __future__ import annotations

import genesis as gs
import torch
from gymnasium import spaces

from dexsuite.options import CamerasOptions, DynamicCamOptions, StaticCamOptions
from dexsuite.utils import get_n_envs
from dexsuite.utils.camera_utils import (
    dynamic_camera_overrides,
    dynamic_camera_targets,
    normalize_quat_wxyz,
    select_dynamic_camera_override,
)
from dexsuite.utils.options_utils import load_config
from dexsuite.utils.space_utils import float_box, frame_to_tensor, int32_box, uint8_box


class CameraSystem:
    """Manage static and dynamic cameras within a Genesis scene.

    Camera creation happens once the scene exists, while dynamic mounts are attached after robot entities are built. Observation spaces mirror the configured modalities and image resolutions.

    Attributes:
        cam_options: Camera configuration object.
        scene: Genesis scene that owns camera entities.
        robot: Robot wrapper used for dynamic camera attachment.
        device: Torch device used for rendered tensors.
    """

    def __init__(
        self,
        scene: gs.Scene,
        robot,
        cam_options: CamerasOptions,
        gui: bool | dict[str, bool] | None = None,
    ) -> None:
        """Initialize the object.

        Args:
        scene: Genesis scene that owns camera entities.
        robot: Robot wrapper used for dynamic camera attachment.
        cam_options: Camera configuration object.
        gui: Global override for camera GUI visualization. Defaults to None.
        """
        self.scene: gs.Scene = scene
        self.robot = robot
        if isinstance(gui, dict):
            if any(not isinstance(name, str) for name in gui):
                raise TypeError("CameraSystem.gui mapping keys must be strings.")
            if any(not isinstance(enabled, bool) for enabled in gui.values()):
                raise TypeError("CameraSystem.gui mapping values must be bools.")
        elif gui is not None and not isinstance(gui, bool):
            raise TypeError("CameraSystem.gui must be a bool, dict, or None.")
        self.gui = gui
        self.cam_options = cam_options

        # camera name -> (gs.Camera, modality_flags)
        self._cams: dict[str, tuple[gs.Camera, dict[str, bool]]] = {}
        # camera name -> (gs.Camera, DynamicCamOptions, hand_ent, root_link)
        self._dyn_specs: dict[
            str,
            tuple[gs.Camera, DynamicCamOptions, gs.Entity, str],
        ] = {}

    @property
    def num_cameras(self) -> int:
        """Return the number of registered cameras."""
        return len(self._cams)

    # ---------------- helpers: enumerate grippers ----------------

    def _all_grippers(self) -> list[tuple[str, gs.Entity, str, object, object]]:
        """Enumerates all grippers present on the robot.

        Detects bimanual setups (robot.left, robot.right) first,
        then falls back to a single-arm setup (robot.hand_ent).

        - ModularRobot: hand_ent is the separate gripper entity; gripper_obj is
          a GripperModel.
        - IntegratedRobot: hand_ent == arm_ent; gripper_obj is the _HandProxy.
        - Bimanual: we detect 'left'/'right' sub-robots automatically.

        Returns:
            A list of tuples, where each tuple contains:
            (label, hand_ent, root_link_name, manip_obj, gripper_obj).
            - label (str): 'left', 'right', or 'arm'.
            - hand_ent (gs.Entity): The entity containing the gripper.
            - root_link_name (str): Name of the gripper's root link.
            - manip_obj (object): The manipulator object (e.g., arm).
            - gripper_obj (object): The gripper object.

        Raises:
            RuntimeError: If no grippers could be found on the robot.
        """
        grippers: list[tuple[str, gs.Entity, str, object, object]] = []

        # Bimanual robot wrapper if present.
        if hasattr(self.robot, "left") and hasattr(self.robot, "right"):
            for side in ("left", "right"):
                side_obj = getattr(self.robot, side)
                if (
                    hasattr(side_obj, "hand_ent")
                    and hasattr(side_obj, "gripper")
                    and hasattr(side_obj.gripper, "root_link")
                ):
                    grippers.append(
                        (
                            side,
                            side_obj.hand_ent,
                            side_obj.gripper.root_link,
                            side_obj.manip,
                            side_obj.gripper,
                        ),
                    )

        # Single-arm (modular or integrated) at top-level
        elif (
            hasattr(self.robot, "hand_ent")
            and hasattr(self.robot, "gripper")
            and hasattr(self.robot.gripper, "root_link")
        ):
            grippers.append(
                (
                    "arm",
                    self.robot.hand_ent,
                    self.robot.gripper.root_link,
                    getattr(self.robot, "manip", None),
                    self.robot.gripper,
                ),
            )

        if not grippers:
            raise RuntimeError(
                "CameraSystem: could not find any gripper(s) on the robot.",
            )
        return grippers

    def _resolve_dynamic_spec(
        self,
        entry: DynamicCamOptions | str,
        manip_obj,
        gripper_obj,
    ) -> DynamicCamOptions:
        """Resolves a dynamic camera entry into a concrete DynamicCamOptions.

        If entry is already a DynamicCamOptions, it is returned directly. If
        entry is a string, it is treated as a preset name and looked up in the
        cameras.yaml defaults.

        The selection logic for presets is:
        1. If modular gripper (has _registry_name): Try overrides[gripper_id].
        2. If integrated (no _registry_name): Try overrides[manip_id].
        3. Fallback: Use preset["default"].

        The selected block may carry its own `fov` and `res`; otherwise the
        preset-level values apply.

        Args:
            entry: A DynamicCamOptions instance or a string preset name.
            manip_obj: The manipulator object (for integrated matching).
            gripper_obj: The gripper object (for modular matching).

        Returns:
            A concrete DynamicCamOptions instance with resolved settings.

        Raises:
            TypeError: If entry is not a DynamicCamOptions or a string.
            ValueError: If the preset string is not found or if a
                matching override or 'default' block is missing in the preset.
        """
        if isinstance(entry, DynamicCamOptions):
            return entry  # already concrete

        if not isinstance(entry, str):
            raise TypeError(
                "Dynamic camera entry must be a DynamicCamOptions or a preset string.",
            )

        preset = self._get_dynamic_preset(entry)
        default = preset.get("default") or {}
        if not isinstance(default, dict):
            raise TypeError(f"Dynamic preset '{entry}' default must be a mapping.")

        overrides = dynamic_camera_overrides(preset, entry)
        choice = select_dynamic_camera_override(
            entry=entry,
            default=default,
            overrides=overrides,
            manip_obj=manip_obj,
            gripper_obj=gripper_obj,
        )
        merged = {**default, **choice}
        pos = tuple(merged.get("pos_offset", (0.0, 0.0, 0.0)))
        quat = tuple(merged.get("quat_offset", (1.0, 0.0, 0.0, 0.0)))
        fov = float(merged.get("fov", preset.get("fov", 60.0)))
        res = tuple(merged.get("res", preset.get("res", (224, 224))))
        gui = choice.get("gui", default.get("gui", preset.get("gui", False)))

        return DynamicCamOptions(
            link=merged.get("link"),
            pos_offset=pos,
            quat_offset=quat,
            fov=fov,
            res=res,
            gui=gui,
            near=float(merged.get("near", preset.get("near", 0.1))),
            far=float(merged.get("far", preset.get("far", 20.0))),
        )

    @staticmethod
    def _get_dynamic_preset(entry: str) -> dict:
        camera_defaults = load_config("cameras")
        dynamic_defaults = camera_defaults.get("dynamic", {})
        preset = dynamic_defaults.get(entry)
        if not isinstance(preset, dict):
            raise TypeError(
                f"Dynamic preset '{entry}' must resolve to a mapping in cameras.yaml.",
            )
        return preset

    @staticmethod
    def _dynamic_overrides(preset: dict, entry: str) -> dict[str, object]:
        return dynamic_camera_overrides(preset, entry)

    # ---------------- lifecycle ----------------
    def _camera_gui(
        self,
        name: str,
        spec: StaticCamOptions | DynamicCamOptions,
    ) -> bool:
        """Return the Genesis GUI flag for an individual camera spec."""
        if isinstance(self.gui, bool):
            return self.gui
        if isinstance(self.gui, dict):
            return self.gui.get(name, spec.gui)
        return spec.gui

    def create(self) -> None:
        """Create all static and dynamic cameras.

        Static cameras are created exactly as defined in cam_options.static.

        Dynamic cameras are attached to discovered grippers:
        - Single-arm: each dynamic entry creates exactly one camera.
        - Bimanual:
          - Generic names (for example, "wrist") expand to "left_wrist" and
            "right_wrist".
          - Side-specific names using a left_ or right_ prefix (or _left/_right
            suffix) are created only for the matching side.
        """
        enabled = set(self.cam_options.modalities)
        flags = {
            "depth": "depth" in enabled,
            "segmentation": "segmentation" in enabled,
            "normal": "normal" in enabled,
        }

        # Static
        for name, spec in self.cam_options.static.items():
            if not isinstance(spec, StaticCamOptions):
                raise TypeError("Static entries must be StaticCamOptions.")
            cam = self.scene.add_camera(
                pos=tuple(spec.pos),
                lookat=tuple(spec.lookat),
                fov=float(spec.fov),
                res=tuple(spec.res),
                GUI=self._camera_gui(name, spec),
                near=float(spec.near),
                far=float(spec.far),
            )
            self._cams[name] = (cam, flags)

        # Dynamic
        grippers = self._all_grippers()
        by_label: dict[str, tuple[gs.Entity, str, object, object]] = {
            label: (hand_ent, root_link, manip_obj, gripper_obj)
            for (label, hand_ent, root_link, manip_obj, gripper_obj) in grippers
        }
        is_bimanual = "left" in by_label and "right" in by_label

        for requested_name, entry in self.cam_options.dynamic.items():
            targets = dynamic_camera_targets(
                requested_name,
                is_bimanual=is_bimanual,
                single_label=grippers[0][0],
            )

            for name, label in targets:
                if name in self._cams:
                    raise ValueError(f"Duplicate camera name '{name}'.")
                if label not in by_label:
                    raise ValueError(
                        f"Dynamic camera '{name}' targets unknown gripper "
                        f"label '{label}'.",
                    )

                hand_ent, root_link, manip_obj, gripper_obj = by_label[label]
                spec = self._resolve_dynamic_spec(entry, manip_obj, gripper_obj)

                cam = self.scene.add_camera(
                    pos=(0.0, 0.0, 0.0),
                    lookat=(0.0, 0.0, 0.0),
                    fov=float(spec.fov),
                    res=tuple(spec.res),
                    GUI=self._camera_gui(name, spec),
                    near=float(spec.near),
                    far=float(spec.far),
                )
                self._cams[name] = (cam, flags)
                self._dyn_specs[name] = (cam, spec, hand_ent, root_link)

    def mount(self) -> None:
        """Attach dynamic cams to the gripper root_link (per env)."""
        batch_size = get_n_envs()

        for cam, spec, hand_ent, root_link in self._dyn_specs.values():
            link_obj = hand_ent.get_link(root_link)

            px, py, pz = map(float, spec.pos_offset)
            w, x, y, z = normalize_quat_wxyz(spec.quat_offset)

            # link->camera transform
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            transform = [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), px],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), py],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), pz],
                [0.0, 0.0, 0.0, 1.0],
            ]

            if batch_size > 1:
                for _env_id in range(batch_size):
                    cam.attach(link_obj, transform)
            else:
                cam.attach(link_obj, transform)

    @staticmethod
    def _shape_hw(
        res_hw: tuple[int, int],
        channels: int | None,
        *,
        batch_size: int,
    ) -> tuple[int, ...]:
        """Return the observation shape for a camera resolution."""
        width, height = int(res_hw[0]), int(res_hw[1])
        if batch_size > 1:
            return (
                (batch_size, height, width, channels)
                if channels
                else (batch_size, height, width)
            )
        return (height, width, channels) if channels else (height, width)

    @classmethod
    def _camera_space(
        cls,
        res_hw: tuple[int, int],
        *,
        batch_size: int,
        want_depth: bool,
        want_seg: bool,
        want_norm: bool,
    ) -> spaces.Dict:
        """Build the observation-space dictionary for one camera."""
        items: dict[str, spaces.Box] = {
            "rgb": uint8_box(
                0,
                255,
                shape=cls._shape_hw(res_hw, 3, batch_size=batch_size),
            ),
        }
        if want_depth:
            items["depth"] = float_box(
                0.0,
                float("inf"),
                shape=cls._shape_hw(res_hw, None, batch_size=batch_size),
            )
        if want_seg:
            items["segmentation"] = int32_box(
                0,
                2_147_483_647,
                shape=cls._shape_hw(res_hw, None, batch_size=batch_size),
            )
        if want_norm:
            items["normal"] = float_box(
                -1.0,
                1.0,
                shape=cls._shape_hw(res_hw, 3, batch_size=batch_size),
            )
        return spaces.Dict(items)

    def obs_space(self) -> dict[str, spaces.Dict]:
        """Builds the Gymnasium observation space for all cameras.

        The space is a dictionary mapping camera names to spaces.Dict objects.
        Each per-camera dictionary maps modality names (for example, "rgb" and
        "depth") to a spaces.Box.

        The shapes and data types are as follows:

        Shapes (B = number of environments, H = height, W = width):
            - single env (B=1):
                - rgb/normal: (H, W, 3)
                - depth/segmentation: (H, W)
            - batched (B>1):
                - rgb/normal: (B, H, W, 3)
                - depth/segmentation: (B, H, W)

        Dtypes:
            - rgb: uint8 [0..255]
            - depth: float32 [0..+inf)
            - segmentation: int32 [0..INT32_MAX]
            - normal: float32 [-1..1]

        Returns:
            dict[str, spaces.Dict]: Observation space for all cameras and modalities.
        """
        batch_size = get_n_envs()
        enabled = set(self.cam_options.modalities)
        want_depth = "depth" in enabled
        want_seg = "segmentation" in enabled
        want_norm = "normal" in enabled

        cams: dict[str, spaces.Dict] = {}

        for name, spec in getattr(self.cam_options, "static", {}).items():
            if not isinstance(spec, StaticCamOptions):
                continue
            cams[name] = self._camera_space(
                spec.res,
                batch_size=batch_size,
                want_depth=want_depth,
                want_seg=want_seg,
                want_norm=want_norm,
            )

        for name, (_cam, spec, _hand_ent, _root_link) in self._dyn_specs.items():
            cams[name] = self._camera_space(
                spec.res,
                batch_size=batch_size,
                want_depth=want_depth,
                want_seg=want_seg,
                want_norm=want_norm,
            )

        return cams

    def render_all(self) -> dict[str, dict[str, torch.Tensor]]:
        """Renders all requested modalities from all cameras.

        The output shapes come directly from Genesis (no extra massaging).

        Returns:
            A dictionary mapping camera names (str) to results.
            Each result is a dictionary mapping modality name
            (e.g., 'rgb', 'depth') to a torch.Tensor containing
            the rendered data.
        """
        enabled = tuple(self.cam_options.modalities)
        use_depth = "depth" in enabled
        use_seg = "segmentation" in enabled
        use_norm = "normal" in enabled

        rendered: dict[str, dict[str, torch.Tensor]] = {}
        for name, (cam, _spec) in self._cams.items():
            frames = cam.render(depth=use_depth, segmentation=use_seg, normal=use_norm)
            rgb = frames[0]
            depth = frames[1] if len(frames) > 1 else None
            segmentation = frames[2] if len(frames) > 2 else None
            normal = frames[3] if len(frames) > 3 else None

            camera_frames: dict[str, torch.Tensor] = {}
            camera_frames["rgb"] = frame_to_tensor(rgb)
            if use_depth and depth is not None:
                camera_frames["depth"] = frame_to_tensor(depth)
            if use_seg and segmentation is not None:
                camera_frames["segmentation"] = frame_to_tensor(segmentation)
            if use_norm and normal is not None:
                camera_frames["normal"] = frame_to_tensor(normal)
            rendered[name] = camera_frames
        return rendered
