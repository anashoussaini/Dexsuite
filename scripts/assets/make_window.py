"""Double-hung window (Artiverse window/3dw/123e1b3...) -> task/window.

Frame 84.4 x 5.9 x 120 cm, front = -y. Lower sash ``link_2_panel`` slides on ``joint_2_to_4`` (+z,
0 = closed, 0.5356 = fully raised); the upper sash is frozen shut; the latch ``link_1_handle`` turns
about +z on the lower sash. Collision is box primitives (frame jambs/rails, sash slabs). The lower sash
gets an added lift rail (visual + collision box) on the front of its bottom rail: the Artiverse sash is a
flush slab with nothing to push down on.
"""

from _common import Box, JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "window", "3dw", "123e1b3f635350647c9627f5c2a486"

XO, XI, Y, ZO, ZI = 0.4222, 0.3722, 0.0294, 0.6, 0.5581

FRAME_BOXES = [
    Box.from_bounds((-XO, -Y, -ZO), (-XI, Y, ZO)),  # -x jamb
    Box.from_bounds((XI, -Y, -ZO), (XO, Y, ZO)),  # +x jamb
    Box.from_bounds((-XI, -Y, -ZO), (XI, Y, -ZI)),  # bottom rail
    Box.from_bounds((-XI, -Y, ZI), (XI, Y, ZO)),  # top rail
]
LOWER_SASH = [
    Box.from_bounds((-0.3785, -0.0216, -0.5559), (0.3785, 0.0002, 0.0328)),
    Box.from_bounds((-0.0113, -0.0217, 0.0315), (0.0414, -0.0011, 0.0487)),  # latch base
]
LIFT_RAIL = Box(center=(0.0, -0.0216 - 0.015, -0.525), size=(0.30, 0.03, 0.02))  # top face z -0.515, 3 cm proud
UPPER_SASH = [Box.from_bounds((-0.3785, 0.0002, -0.0052), (0.3785, 0.022, 0.5689))]


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "task" / "window",
        "window",
        links={
            "link_4_base": LinkSpec(mass=25.0, collision="primitives", primitives=FRAME_BOXES),
            "link_2_panel": LinkSpec(
                mass=2.0, collision="primitives", primitives=LOWER_SASH,
                visual_primitives=[LIFT_RAIL], visual_rgba=(0.45, 0.33, 0.20, 1.0),
            ),
            "link_3_panel": LinkSpec(mass=2.0, collision="primitives", primitives=UPPER_SASH),
            "link_1_handle": LinkSpec(mass=0.05, collision="hull"),
        },
        joints={
            "joint_2_to_4": JointSpec(lower=0.0, upper=0.5356, effort=200.0, damping=20.0, friction=30.0),
            "joint_3_to_4": JointSpec(type="fixed"),
            "joint_1_to_2": JointSpec(effort=2.0, damping=0.05, friction=0.05),
        },
        attribution=attribution_text(CAT, SRC, MID, "make_window.py"),
    )


if __name__ == "__main__":
    main()
