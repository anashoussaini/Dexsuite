"""Define shared rigid-body component model behavior.

Component numeric parameters are stored as float32 torch tensors on the global
DexSuite device. Scalar gains and limits are broadcast to the model DoF count,
and vector inputs must match that count exactly.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from pathlib import Path

import torch

from dexsuite.utils import get_device


class RigidBodyModel(ABC):
    """Represent a rigid robot component described by a model asset.

    Attributes:
        model_path: Path to the component model file.
        dof: Number of controlled degrees of freedom.
        device: Torch device used for component tensors.
        kp: Proportional gains as a `(dof,)` float32 tensor, or `None`.
        kv: Derivative gains as a `(dof,)` float32 tensor, or `None`.
        force_range: Effort limits as a `(dof,)` float32 tensor, or `None`.
        q_lo: Lower joint limits as a `(dof,)` float32 tensor, or `None`.
        q_hi: Upper joint limits as a `(dof,)` float32 tensor, or `None`.
    """

    model_path: str
    dof: int
    control_dofs_index: list[int] | None = None
    kp: float | Sequence[float] | torch.Tensor | None = None
    kv: float | Sequence[float] | torch.Tensor | None = None
    force_range: float | Sequence[float] | torch.Tensor | None = None
    q_lo: float | Sequence[float] | torch.Tensor | None = None
    q_hi: float | Sequence[float] | torch.Tensor | None = None
    fixed_base: bool = True

    def __init__(self) -> None:
        """Initialize the rigid-body model and validate required class fields."""
        if self.model_path is None or self.dof is None:
            raise NotImplementedError("model_path and dof must be defined")
        if not Path(self.model_path).exists():
            from dexsuite.utils.models_utils import MISSING_ASSET_HINT

            raise FileNotFoundError(f"{type(self).__name__}: {self.model_path} is missing. {MISSING_ASSET_HINT}")

        self.device = get_device()

        self.kp = self._process_param(self.kp, self.dof, self.device, "kp")
        self.kv = self._process_param(self.kv, self.dof, self.device, "kv")
        self.force_range = self._process_param(
            self.force_range,
            self.dof,
            self.device,
            "force_range",
        )
        self.q_lo = self._process_param(self.q_lo, self.dof, self.device, "q_lo")
        self.q_hi = self._process_param(self.q_hi, self.dof, self.device, "q_hi")

    def _process_param(
        self,
        value: float | Sequence[float] | torch.Tensor | None,
        size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor | None:
        """Convert or broadcast a component parameter to a device tensor.

        Args:
            value: Scalar, sequence, tensor, or `None` parameter value.
            size: Required one-dimensional tensor length.
            device: Destination torch device.
            name: Parameter name used in validation errors.

        Returns:
            `(size,)` float32 tensor on `device`, or `None` when no value was supplied.

        Raises:
            TypeError: When `value` has an unsupported type.
            ValueError: When `value` cannot be broadcast to `(size,)`.
        """
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            t = value.to(device)
        elif isinstance(value, (float, int)):
            t = torch.full((size,), float(value), device=device, dtype=torch.float32)
        elif isinstance(value, Sequence):
            t = torch.tensor(value, device=device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for {name}: {type(value)}")

        if t.numel() == 1 and size > 1:
            return t.expand(size)
        if t.shape != (size,):
            raise ValueError(
                f"{name} has wrong shape. Expected ({size},) or scalar, "
                f"but got {t.shape}",
            )
        return t

    def default_qpos(self) -> torch.Tensor:
        """Return the default joint configuration for the component.

        Subclasses may override this when a hardware model has a meaningful
        nominal home pose. The base implementation returns zeros on the
        component device.

        Returns:
            `(dof,)` float32 tensor containing joint positions.
        """
        return torch.zeros(self.dof, device=self.device, dtype=torch.float32)
