"""Top-level aggregator for all environment configuration options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .camera import CamerasOptions
from .layout import LayoutOptions
from .robot import RobotOptions
from .sim import SimOptions

RenderMode = Literal["human", "rgb_array"] | None


@dataclass(slots=True)
class EnvOptions:
    """Aggregate all environment options into a single object.

    This class provides a convenient way to pass all configuration parameters
    to the dexsuite.make function.

    Attributes:
        sim: Simulation-related options, like control frequency.
        layout: Robot placement and workspace options.
        robot: Robot hardware and controller specifications.
        cameras: Camera configurations and rendering modalities.
        render_mode: The rendering mode for the environment, if any.
    """

    sim: SimOptions = field(default_factory=SimOptions)
    layout: LayoutOptions = field(default_factory=LayoutOptions)
    robot: RobotOptions = field(default_factory=RobotOptions)
    cameras: CamerasOptions = field(default_factory=CamerasOptions)
    render_mode: RenderMode = None
