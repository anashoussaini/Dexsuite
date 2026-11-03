"""Expose the UMI gripper model.

Submodules:
    umi: UMI parallel-jaw gripper with tactile fingertip pads.
"""

from .umi import Umi

FrankaUmiTacticle = Umi  # the former class name
