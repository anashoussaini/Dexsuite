"""Import bundled gripper model registrations.

Subpackages:
    ability: Ability Hand gripper model.
    allegro: Allegro Hand gripper models.
    barrett: BarrettHand gripper model.
    dclaw: DClaw gripper model.
    inspire: Inspire Hand gripper model.
    leap: LEAP Hand gripper models.
    robotiq: Robotiq 2F-85 gripper model.
    schunk: Schunk SVH gripper model.
    shadow: Shadow Hand gripper models.
    sharpa: Sharpa hand gripper models.
    umi: UMI gripper with tactile fingertip pads.
"""

import pkgutil
from importlib import import_module

for module_info in pkgutil.walk_packages(__path__, f"{__name__}."):
    import_module(module_info.name)
