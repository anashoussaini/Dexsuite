"""Shared low-frequency target noise for automated data collection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass
class TargetNoiseState:
    """Hold and resample smooth target perturbations for one episode.

    The perturbation is applied to scripted Cartesian targets before the
    controller converts them into actions. Holding each sampled offset for
    several control steps gives trajectory diversity without adding white-noise
    jitter to every `env.step(action)` command.
    """

    xy_std: float
    z_std: float
    hold_steps: int
    max_xy: float
    max_z: float
    phases: tuple[str, ...]
    generator: torch.Generator | None = None
    generator_seed: int | None = None
    generator_device: str | None = None
    offset: torch.Tensor | None = None
    remaining: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, episode: int) -> TargetNoiseState:
        """Build a per-episode noise state from parsed CLI arguments."""
        seed = getattr(args, "target_noise_seed", None)
        xy_std = max(float(getattr(args, "target_noise_xy_std", 0.0)), 0.0)
        z_std = max(float(getattr(args, "target_noise_z_std", 0.0)), 0.0)
        phases = tuple(
            part.strip().lower()
            for part in str(getattr(args, "target_noise_phases", "")).split(",")
            if part.strip()
        )
        return cls(
            xy_std=xy_std,
            z_std=z_std,
            hold_steps=max(int(getattr(args, "target_noise_hold_steps", 12)), 1),
            max_xy=max(float(getattr(args, "target_noise_max_xy", 0.015)), 0.0),
            max_z=max(float(getattr(args, "target_noise_max_z", 0.010)), 0.0),
            phases=phases,
            generator_seed=None if seed is None else int(seed) + int(episode),
        )

    @property
    def enabled(self) -> bool:
        """Return whether this state will perturb selected phases."""
        return (self.xy_std > 0.0 or self.z_std > 0.0) and bool(self.phases)

    def apply(self, target: torch.Tensor, phase: str, *, final: bool = False) -> torch.Tensor:
        """Return a perturbed target when `phase` is selected.

        Args:
            target: Unperturbed Cartesian target.
            phase: Name of the active scripted phase.
            final: Whether this target is the final sample of a segment.

        Returns:
            The original target or a target plus a held offset.
        """
        if not self.enabled or final or not self._phase_enabled(phase):
            return target
        if self.offset is None or self.remaining <= 0:
            self.offset = self._sample_offset(target)
            self.remaining = self.hold_steps
        self.remaining -= 1
        return (target + self.offset.to(dtype=target.dtype, device=target.device)).contiguous()

    def _phase_enabled(self, phase: str) -> bool:
        phase_name = phase.lower()
        return "all" in self.phases or any(token in phase_name for token in self.phases)

    def _sample_offset(self, target: torch.Tensor) -> torch.Tensor:
        generator = self._generator_for(target.device)
        sample = torch.randn(
            3,
            dtype=torch.float32,
            device=target.device,
            generator=generator,
        )
        sample[0:2] *= self.xy_std
        sample[2] *= self.z_std
        sample[0:2] = sample[0:2].clamp(-self.max_xy, self.max_xy)
        sample[2] = sample[2].clamp(-self.max_z, self.max_z)
        return sample

    def _generator_for(self, device: torch.device) -> torch.Generator | None:
        if self.generator_seed is None:
            return None
        device_name = str(device)
        if self.generator is None or self.generator_device != device_name:
            self.generator = torch.Generator(device=device_name)
            self.generator.manual_seed(self.generator_seed)
            self.generator_device = device_name
        return self.generator


def add_target_noise_args(parser: argparse.ArgumentParser) -> None:
    """Register shared target-noise CLI flags."""
    parser.add_argument(
        "--target-noise-xy-std",
        type=float,
        default=0.0,
        help="Low-frequency target noise stddev for XY during selected phases, in meters.",
    )
    parser.add_argument(
        "--target-noise-z-std",
        type=float,
        default=0.0,
        help="Low-frequency target noise stddev for Z during selected phases, in meters.",
    )
    parser.add_argument(
        "--target-noise-hold-steps",
        type=int,
        default=12,
        help="Number of control steps to hold each sampled target-noise offset.",
    )
    parser.add_argument(
        "--target-noise-max-xy",
        type=float,
        default=0.015,
        help="Absolute clamp for target-noise XY offset, in meters.",
    )
    parser.add_argument(
        "--target-noise-max-z",
        type=float,
        default=0.010,
        help="Absolute clamp for target-noise Z offset, in meters.",
    )
    parser.add_argument(
        "--target-noise-phases",
        default="lift,transport,retreat",
        help="Comma-separated phase-name substrings where target noise is active.",
    )
    parser.add_argument(
        "--target-noise-seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic target-noise offsets.",
    )
