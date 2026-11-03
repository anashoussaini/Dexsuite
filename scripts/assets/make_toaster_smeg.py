"""SMEG two-slot toaster (Artiverse toaster/objaverse/37e9755...) -> kitchen/toaster_smeg.

Body collision is seven boxes so the two slots stay open (unscaled metres, link frame): interior
floor z 0.0233, slot walls y = +-0.0286 (outer) / +-0.0072 (divider), slot x [-0.042, 0.0503],
rim z 0.0602. The -x end block is split so the lever knob slit (shell face x -0.0842) stays free.
Lever: prismatic +z, rest 0 = up, pressed at -0.054.
"""

from _common import Box, JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "toaster", "objaverse", "37e9755326084ef294581d17acdeb4c4"

X0, X1 = -0.0976, 0.0992  # body
Y = 0.0598
Z0, Z1 = -0.0618, 0.0602  # bottom, rim
SX0, SX1 = -0.042, 0.0503  # slot span
ZF = 0.0233  # slot floor
YW, YD = 0.0286, 0.0072  # outer slot wall, divider half-width
X_SHELL = -0.0842  # chrome end face at knob height (z 0.0495..0.0618); the knob slit is outside it

BASE_BOXES = [
    Box.from_bounds((SX0, -Y, Z0), (SX1, Y, ZF)),  # floor block under the slots
    Box.from_bounds((SX1, -Y, Z0), (X1, Y, Z1)),  # +x end block (dial side)
    Box.from_bounds((X0, -Y, Z0), (SX0, Y, 0.0)),  # -x end block, low
    Box.from_bounds((X_SHELL, -Y, 0.0), (SX0, Y, Z1)),  # -x end block, high (knob slit free)
    Box.from_bounds((SX0, YW, ZF), (SX1, Y, Z1)),  # +y side wall
    Box.from_bounds((SX0, -Y, ZF), (SX1, -YW, Z1)),  # -y side wall
    Box.from_bounds((SX0, -YD, ZF), (SX1, YD, Z1)),  # divider
]


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "kitchen" / "toaster_smeg",
        "toaster_smeg",
        links={
            "link_1_base": LinkSpec(mass=1.5, collision="primitives", primitives=BASE_BOXES),
            "link_4_rotary_dial": LinkSpec(mass=0.02, collision="hull"),
            "link_5_lever": LinkSpec(mass=0.02, collision="hull"),
        },
        joints={
            "joint_5_to_1": JointSpec(lower=-0.054, upper=0.0, effort=100.0, damping=2.0, friction=2.0),
            "joint_4_to_1": JointSpec(effort=5.0, damping=0.05, friction=0.05),
        },
        attribution=attribution_text(CAT, SRC, MID, "make_toaster_smeg.py"),
    )


if __name__ == "__main__":
    main()
