"""Pliers (Artiverse pliers/3dw/ee3229d...) -> tools/pliers.

11.3 x 19.1 x 2.7 cm lying flat (thin axis z). Root is the pivot screw with two legs and a jaw. The
legs' revolute joints are frozen: the task only grasps and places the pliers, and the two legs' CoACD
pieces overlap at the pivot (Artiverse's closed pose), which as separate bodies produced permanent
self-collision kicks. Legs are CoACD-decomposed (bent bars).
"""

from _common import JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "pliers", "3dw", "ee3229daab48b2f3bfa59e044d401be9"


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "tools" / "pliers",
        "pliers",
        links={
            "link_5_screw": LinkSpec(mass=0.01, collision="none"),  # sits inside the legs
            "link_6_leg": LinkSpec(mass=0.10, collision="coacd", coacd_threshold=0.04, coacd_max_hulls=6),
            "link_8_handle": LinkSpec(mass=0.10, collision="coacd", coacd_threshold=0.04, coacd_max_hulls=6),
            "link_7_jaw": LinkSpec(mass=0.03, collision="merged_hull"),
        },
        joints={
            "joint_6_to_5": JointSpec(type="fixed"),
            "joint_8_to_5": JointSpec(type="fixed"),
        },
        attribution=attribution_text(CAT, SRC, MID, "make_pliers.py"),
    )


if __name__ == "__main__":
    main()
