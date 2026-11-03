"""Trigger spray bottle (Artiverse spray_bottle/objaverse_xl/745b733...) -> tools/spray_bottle_blue.

Bottle 9.2 x 7.8 x 25 cm (z [-0.125, 0.125]). The head's screw joint and the nozzle joint are frozen;
only the trigger ``joint_2_to_3`` moves (0 = released, 0.3927 = pulled). Nozzle at
(-0.0358, -0.0182, 0.1182) in the container frame, spraying along (-0.887, -0.462, 0).

The container is a full bottle: 0.9 kg with its centre of mass 30 % up from the base (z -0.06, the
hull's is -0.034) so the standing bottle survives nudges; ~0.5 kg total once the env scales it by 0.8.
"""

from _common import JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "spray_bottle", "objaverse_xl", "745b7335cb77ce1b58ec5637ba29072a94d0337acb5185ecac7b56b9d1242808"


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "tools" / "spray_bottle_blue",
        "spray_bottle_blue",
        links={
            "link_1_container": LinkSpec(mass=0.9, collision="merged_hull", com=(None, None, -0.06)),
            "link_3_head": LinkSpec(mass=0.05, collision="merged_hull"),
            "link_2_handle": LinkSpec(mass=0.02, collision="merged_hull"),
            "link_4_nozzle": LinkSpec(mass=0.005, collision="merged_hull"),
        },
        joints={
            "joint_3_to_1": JointSpec(type="fixed"),
            "joint_4_to_3": JointSpec(type="fixed"),
            "joint_2_to_3": JointSpec(lower=0.0, upper=0.3927, effort=5.0, damping=0.02, friction=0.01),
        },
        attribution=attribution_text(CAT, SRC, MID, "make_spray_bottle_blue.py"),
    )


if __name__ == "__main__":
    main()
