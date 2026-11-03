"""Aerosol spray can with a push head (Artiverse aerosol_can/objaverse/ef58952...) -> tools/aerosol_can.

Container o 9.6 x 19.7 cm (z [-0.118, 0.079]); head prismatic along -z, rest 0, pressed at +0.009.
The head meshes are widened 1.3x about the can axis (o 2.4 -> 3.2 cm, same height) so a finger presses it;
nozzle outlet on the head at ~(0, -0.0117, 0.104), spraying toward -y (link frame).
"""

from _common import JointSpec, LinkSpec, OBJECTS_ROOT, attribution_text, find_model_urdf, import_model

CAT, SRC, MID = "aerosol_can", "objaverse", "ef58952aec3b4a93a514f5fff5607d1a"


def main() -> None:
    import_model(
        find_model_urdf(CAT, SRC, MID),
        OBJECTS_ROOT / "tools" / "aerosol_can",
        "aerosol_can",
        links={
            "link_1_container": LinkSpec(mass=2.31, collision="merged_hull", com=(None, None, -0.045)),  # 0.79 kg at the env's 0.7 scale, COM below the hull centroid (-0.026)
            "link_2_head": LinkSpec(mass=0.02, collision="merged_hull", mesh_scale=(1.3, 1.3, 1.0)),
        },
        joints={"joint_2_to_1": JointSpec(lower=0.0, upper=0.009, effort=20.0, damping=2.0, friction=0.2)},
        attribution=attribution_text(CAT, SRC, MID, "make_aerosol_can.py"),
    )


if __name__ == "__main__":
    main()
