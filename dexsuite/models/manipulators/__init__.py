"""Import bundled manipulator model registrations.

Submodules are imported eagerly so registration decorators run at import time.

Subpackages:
    aloha_tactile: ALOHA tactile manipulator model.
    franka: Franka Panda manipulator models.
    franka_fr3: Franka FR3 manipulator model.
    gen3: Kinova Gen3 manipulator model.
    gen3_lite: Kinova Gen3 Lite manipulator model.
    kuka: Kuka LBR iiwa manipulator model.
    link6: Link6 manipulator model.
    lite6: UFactory Lite 6 manipulator model.
    sawyer: Rethink Robotics Sawyer manipulator model.
    so_100: SO-100 manipulator model.
    trossen_vx250s: Trossen WX250S manipulator model.
    trossen_vx300s: Trossen VX300S manipulator model.
    ur10: Universal Robots UR10e manipulator model.
    ur5: Universal Robots UR5e manipulator model.
    z1: Unitree Z1 manipulator model.
"""

import pkgutil
from importlib import import_module

for module_info in pkgutil.walk_packages(__path__, f"{__name__}."):
    import_module(module_info.name)
