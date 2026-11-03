"""Expose Franka manipulator models.

Submodules:
    franka: Franka Panda arm-only model.
    franka_with_gripper: Franka Panda integrated-gripper model.
"""

from .franka import FrankaArm  # triggers decorator
from .franka_with_gripper import FrankaWithGripper
