"""User-facing simulation configuration options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimOptions:
    """User-configurable simulation parameters.

    Attributes:
        control_hz: The frequency (in Hz) at which control actions are applied.
        physics_substeps: Override the task's integration substeps, preserving
            its physical timestep and the control frequency; None uses task defaults.
        performance_mode: A flag to enable or disable performance-optimizing
            settings within the simulation.
        precision: Genesis floating-point precision, fixed for the process.
    """

    control_hz: int = 20
    performance_mode: bool = False
    n_envs: int = 1
    physics_substeps: int | None = None
    precision: str = "32"

    def __post_init__(self) -> None:
        """Validate the control frequency."""
        if self.precision not in ("32", "64"):
            raise ValueError("Sim.precision must be '32' or '64'.")
        if self.physics_substeps is not None and (isinstance(self.physics_substeps,bool) or not isinstance(self.physics_substeps,int) or self.physics_substeps<1):
            raise ValueError("Sim.physics_substeps must be a positive integer or None.")
        if self.control_hz <= 0:
            raise ValueError("Sim.control_hz must be a positive integer.")
        if self.n_envs <= 0:
            raise ValueError("Sim.n_envs must be >= 1.")
