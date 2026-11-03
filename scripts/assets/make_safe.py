"""Safe with a lever handle, door and dial (Artiverse safe/objaverse_xl/0e5a638...) -> task/safe.

Body 35.4 x 45.7 x 60 cm, front = -y, door hinge at +x (axis +z, opens toward -y). Collision: hollow
box primitives for the body (cavity x +-0.146, y [-0.207, 0.165], z [-0.181, 0.279]), box shelves,
box door slab, cylinder dial, CoACD pieces for the three-spoke handle (the only non-convex part).
The source materials are flat grey; the body/door are recoloured steel blue, the handle brass, the dial silver.
"""

import math

from _common import Box, Cylinder, JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "safe", "objaverse_xl", "0e5a6386402c13fddcca2730fab2859c35cface4794b6cee0046eebc1e3ae38d"

XO, YF, YB, ZO = 0.1769, -0.2071, 0.2501, 0.30  # outer half-x, front y, back y, half-z
XI, YI, ZT, ZB = 0.146, 0.165, 0.279, -0.181  # cavity half-x, cavity back y, cavity top z, cavity floor z
DOOR_ORIGIN = (0.14182338, -0.20319238, 0.04162523)
DIAL_ORIGIN = (0.14182338 - 0.13768907, -0.20319238 - 0.00457467, 0.04162523 - 0.13159613)

BASE_BOXES = [
    Box.from_bounds((-XO, YI, -ZO), (XO, YB, ZO)),  # back wall
    Box.from_bounds((-XO, YF, -ZO), (-XI, YI, ZO)),  # -x jamb
    Box.from_bounds((XI, YF, -ZO), (XO, YI, ZO)),  # +x jamb
    Box.from_bounds((-XI, YF, ZT), (XI, YI, ZO)),  # top
    Box.from_bounds((-XI, YF, -ZO), (XI, YI, ZB)),  # bottom / plinth
]
SHELF_TOP = Box.from_bounds((-0.1667, -0.1492, 0.0949), (0.1667, 0.2315, 0.1025))
SHELF_BOT = Box.from_bounds((-0.1667, -0.1492, -0.0318), (0.1667, 0.2315, -0.0201))


def _door_box() -> Box:
    lo = (-0.1302 - DOOR_ORIGIN[0], -0.195 - DOOR_ORIGIN[1], -0.1798 - DOOR_ORIGIN[2])
    hi = (0.1605 - DOOR_ORIGIN[0], -0.154 - DOOR_ORIGIN[1], 0.2557 - DOOR_ORIGIN[2])
    return Box.from_bounds(lo, hi)


def _hub_cylinder() -> Cylinder:
    """Handle boss (2_door__lambert2SG): x [-0.0523, 0.0257], y [-0.2234, -0.1978], z [0.0818, 0.1598], axis y."""
    cx, cy, cz = 0.5 * (-0.0523 + 0.0257), 0.5 * (-0.2234 - 0.1978), 0.5 * (0.0818 + 0.1598)
    return Cylinder(
        center=(cx - DOOR_ORIGIN[0], cy - DOOR_ORIGIN[1], cz - DOOR_ORIGIN[2]),
        radius=0.5 * (0.0257 + 0.0523),
        length=0.2234 - 0.1978,
        rpy=(math.pi / 2, 0.0, 0.0),
    )


def _dial_cylinder() -> Cylinder:
    cx, cy, cz = 0.5 * (-0.0358 + 0.0437), 0.5 * (-0.2229 - 0.1919), 0.5 * (-0.1296 - 0.0501)
    return Cylinder(
        center=(cx - DIAL_ORIGIN[0], cy - DIAL_ORIGIN[1], cz - DIAL_ORIGIN[2]),
        radius=0.5 * (0.0437 + 0.0358),
        length=0.2229 - 0.1919,
        rpy=(math.pi / 2, 0.0, 0.0),  # cylinder axis -> link y (door normal)
    )


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "task" / "safe",
        "safe",
        links={
            "link_1_base": LinkSpec(mass=60.0, collision="primitives", primitives=BASE_BOXES),
            "link_5_shelf": LinkSpec(mass=1.0, collision="primitives", primitives=[SHELF_TOP]),
            "link_6_shelf": LinkSpec(mass=1.0, collision="primitives", primitives=[SHELF_BOT]),
            "link_2_door": LinkSpec(mass=12.0, collision="primitives", primitives=[_door_box(), _hub_cylinder()]),
            "link_3_handle": LinkSpec(mass=0.8, collision="coacd", coacd_threshold=0.03, coacd_max_hulls=10),
            "link_4_dial_lock": LinkSpec(mass=0.15, collision="primitives", primitives=[_dial_cylinder()]),
        },
        joints={
            "joint_2_to_1": JointSpec(effort=200.0, damping=5.0, friction=3.0),
            "joint_3_to_2": JointSpec(effort=50.0, damping=0.2, friction=0.3),
            "joint_4_to_2": JointSpec(effort=5.0, damping=0.02, friction=0.02),
        },
        attribution=attribution_text(CAT, SRC, MID, "make_safe.py"),
        mtl_colors={"Safe": (0.30, 0.38, 0.48), "lambert2SG": (0.72, 0.74, 0.78)},
        mtl_colors_by_file={"3_handle": (0.82, 0.64, 0.24), "4_dial_lock": (0.86, 0.87, 0.90)},
    )


if __name__ == "__main__":
    main()
