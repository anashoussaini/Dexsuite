"""Hinged jewelry box (Artiverse jewelry_box/3dw/b392fb6...) -> task/jewelry_box.

Box 25 x 15.6 x 12.7 cm; cavity x +-0.1094, y +-0.0625, floor z -0.0479, rim z 0.0459 (link frame).
Lid hinge at the +y back edge (axis -x), 0 = closed, pi/2 = open. Collision: floor + four wall boxes
and a lid slab.
"""

from _common import Box, JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "jewelry_box", "3dw", "b392fb6d154df66bd1fba39ed6bb3f3f"

XO, YO, XI, YI = 0.125, 0.0781, 0.1094, 0.0625
Z0, ZF, ZR, ZT = -0.0635, -0.0479, 0.0459, 0.0635
LID_ORIGIN = (-0.00248282, 0.06093833, 0.04589844)

BASE_BOXES = [
    Box.from_bounds((-XO, -YO, Z0), (XO, YO, ZF)),  # floor
    Box.from_bounds((-XO, -YO, ZF), (-XI, YO, ZR)),  # -x wall
    Box.from_bounds((XI, -YO, ZF), (XO, YO, ZR)),  # +x wall
    Box.from_bounds((-XI, -YO, ZF), (XI, -YI, ZR)),  # -y wall
    Box.from_bounds((-XI, YI, ZF), (XI, YO, ZR)),  # +y wall (hinge side)
]
LID_BOX = Box.from_bounds(
    (-XO - LID_ORIGIN[0], -YO - LID_ORIGIN[1], ZR - LID_ORIGIN[2]),
    (XO - LID_ORIGIN[0], YO - LID_ORIGIN[1], ZT - LID_ORIGIN[2]),
)


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "task" / "jewelry_box",
        "jewelry_box",
        links={
            "link_1_base": LinkSpec(mass=1.0, collision="primitives", primitives=BASE_BOXES),
            "link_2_lid": LinkSpec(mass=0.25, collision="primitives", primitives=[LID_BOX]),
        },
        joints={"joint_2_to_1": JointSpec(effort=20.0, damping=0.1, friction=0.2)},
        attribution=attribution_text(CAT, SRC, MID, "make_jewelry_box.py"),
    )


if __name__ == "__main__":
    main()
