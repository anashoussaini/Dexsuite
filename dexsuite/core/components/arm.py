"""Define manipulator model base classes.

Manipulator poses use meters and radians, with quaternions stored in WXYZ order
where present. Modular arms load adapter catalogs lazily, while integrated arms
describe their built-in gripper joints in the same entity DoF vector.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import torch

from dexsuite.core.components.base import RigidBodyModel
from dexsuite.utils import load_component_yaml


class ArmModel(RigidBodyModel):
    """Represent shared metadata for robotic manipulator models.

    Subclasses define the model asset, DoF count, end-effector link, and root
    link used by controllers and robot factories. Modular arms may also provide
    a mount adapter YAML file next to the model class.

    Attributes:
        end_link: End-effector link name used for IK and OSC.
        root_link: Link name recommended for anchoring or placement.
        home_q: Nominal arm joint configuration as a `(dof,)` float32 tensor.
        kp: Per-joint proportional gains as a `(dof,)` float32 tensor, or `None`.
        kv: Per-joint derivative gains as a `(dof,)` float32 tensor, or `None`.
        force_range: Per-joint effort limits as a `(dof,)` float32 tensor, or `None`.
        q_lo: Lower joint limits as a `(dof,)` float32 tensor, or `None`.
        q_hi: Upper joint limits as a `(dof,)` float32 tensor, or `None`.
    """

    end_link: str
    root_link: str
    home_q: torch.Tensor

    kp: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    force_range: torch.Tensor | None = None
    q_lo: torch.Tensor | None = None
    q_hi: torch.Tensor | None = None

    _adapter_yaml: str | None = None

    @classmethod
    def _load_adapter_yaml(cls) -> dict:
        """Locate and load the mount-adapter YAML file.

        The method searches for the YAML file in the following order:
            1. The explicit path defined in _adapter_yaml.
            2. A file named <registry>_adapters.yaml if _registry_name is set.
            3. A file named <module>_adapters.yaml next to the class definition.

        Returns:
            Parsed adapter specification, or an empty dictionary when no file is found.
        """
        model_file = Path(inspect.getfile(cls))

        if cls._adapter_yaml:
            adapter_path = Path(cls._adapter_yaml)
            if not adapter_path.is_absolute():
                adapter_path = model_file.with_name(cls._adapter_yaml)
            if adapter_path.exists():
                return load_component_yaml(adapter_path)

        registry_name = getattr(cls, "_registry_name", None)
        if registry_name:
            adapter_path = model_file.with_name(
                f"{registry_name.lower()}_adapters.yaml",
            )
            if adapter_path.exists():
                return load_component_yaml(adapter_path)

        adapter_path = model_file.with_name(f"{model_file.stem}_adapters.yaml")
        if adapter_path.exists():
            return load_component_yaml(adapter_path)

        return {}


class ModularManipulatorModel(ArmModel):
    """Represent an arm that accepts separate gripper attachments.

    Adapter metadata maps gripper registry keys to mount poses and optional
    mount geometry. The adapter catalog is read lazily from YAML so importing
    model classes does not perform file I/O.

    Attributes:
        has_builtin_gripper: Whether the manipulator asset includes a built-in gripper.
    """

    has_builtin_gripper = False

    @property
    def adapters(self) -> dict[str, dict]:
        """Return the adapter catalog for attaching grippers.

        The catalog maps gripper registry keys to their corresponding adapter
        specifications. The associated YAML file is loaded lazily on first
        access and cached on the instance to avoid I/O at import time.

        Returns:
            Mapping from gripper registry key to adapter specification.
        """
        cache_attr = "_adapters_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, self._load_adapter_yaml())
        return getattr(self, cache_attr)


class IntegratedManipulatorModel(ArmModel):
    """Represent an arm whose gripper is part of the same model asset.

    Built-in gripper DoFs are expressed in full-entity DoF indices. If explicit
    indices are omitted, the gripper is assumed to occupy the final contiguous
    DoFs in the model.

    Attributes:
        has_builtin_gripper: Whether the manipulator asset includes a built-in gripper.
        builtin_gripper_dof: Number of DoFs in the built-in gripper.
        builtin_gripper_root_link: Link whose frame expresses the built-in gripper root.
        builtin_gripper_control_dofs_index: Full-entity DoF indices for gripper joints, or `None` to use trailing DoFs.
        builtin_gripper_q_lo: Lower gripper joint limits as a `(builtin_gripper_dof,)` float32 tensor, or `None`.
        builtin_gripper_q_hi: Upper gripper joint limits as a `(builtin_gripper_dof,)` float32 tensor, or `None`.
        builtin_gripper_home_q: Nominal gripper configuration as a `(builtin_gripper_dof,)` float32 tensor, or `None`.
        builtin_gripper_tcp_pose: TCP pose `[x, y, z, roll, pitch, yaw]` in the gripper root frame, or `None`.
        tactile: Whether this integrated manipulator exposes tactile pads.
        builtin_tactile_links: Link names to expose as tactile pads, or `None`.
        builtin_tactile_grid_size: Genesis force-grid resolution for each tactile pad, or `None`.
        builtin_tactile_point_distance: Spacing in meters between tactile sample points, or `None`.
    """

    has_builtin_gripper = True

    builtin_gripper_dof: int
    builtin_gripper_root_link: str
    builtin_gripper_control_dofs_index: Sequence[int] | None = None

    builtin_gripper_q_lo: torch.Tensor | None = None
    builtin_gripper_q_hi: torch.Tensor | None = None
    builtin_gripper_home_q: torch.Tensor | None = None

    builtin_gripper_tcp_pose: torch.Tensor | None = None

    tactile: bool = False
    builtin_tactile_links: Sequence[str] | None = None
    builtin_tactile_grid_size: tuple[int, int, int] | None = None
    builtin_tactile_point_distance: float | None = None

    @property
    def gripper_dofs_index(self) -> tuple[int, ...]:
        """Return built-in gripper DoF indices in the full entity DoF vector.

        Integrated models should define builtin_gripper_control_dofs_index when
        the gripper joints are not the final contiguous DoFs.

        Returns:
            A tuple of full-entity DoF indices for the built-in gripper.

        Raises:
            AttributeError: If builtin_gripper_dof is not defined.
            ValueError: If configured indices do not match the gripper DoF count.
        """
        dof = getattr(self, "builtin_gripper_dof", None)
        if dof is None:
            raise AttributeError(
                "integrated manipulator must define builtin_gripper_dof",
            )
        dof = int(dof)
        explicit = getattr(self, "builtin_gripper_control_dofs_index", None)
        if explicit is None:
            return tuple(range(int(self.dof) - dof, int(self.dof)))

        indices = tuple(int(i) for i in explicit)
        if len(indices) != dof:
            raise ValueError(
                "builtin_gripper_control_dofs_index length "
                f"{len(indices)} does not match builtin_gripper_dof {dof}.",
            )
        return indices
