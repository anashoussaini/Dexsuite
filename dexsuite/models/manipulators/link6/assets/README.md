# Kinova Gen3 Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 2.3.3 or later.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

## Overview

This package contains a simplified robot description (MJCF) of the [Kinova Link6](https://www.kinovarobotics.com/product/link-6-cobot) developed by [Kinova Robotics](https://www.kinovarobotics.com/). It is derived from the link6_nominal.xacro file **(related GitHub repo not publicly shared yet and will be added later)**.

**Photo to be replaced by Link6 simulation photo later**
<p float="left">
  <img src="gen3.png" width="400">
</p>

## MJCF derivation steps

1. Converted xacro to URDF: `xacro link6_nominal.xacro -o link6.urdf`
2. Added the following inside the URDF's `<robot>` clause:

```xml
<mujoco>
  <compiler meshdir="assets" discardvisual="false" fusestatic="false" balanceinertia="true"/>
</mujoco>
```

Note `fusestatic="false"` is needed so that base_link is preserved. Also, `balanceinertia="true"` is used to fix the inertia tensor for the `upper_wrist_link`.

3. Converted to MJCF:

```python
import mujoco
model = mujoco.MjModel.from_xml_path('link6.urdf')
mujoco.mj_saveLastXML('link6.xml', model)
print("Wrote link6.xml")
```

4. Cleaned up the MJCF file as follows:
    - Added option: `<option integrator="implicitfast"/>`.
    - Replaced `type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.529412 0.529412 0.529412 1"` and `type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.196078 0.196078 0.196078 1"` with `class="visual"` and added the class to defaults.
    - Added the collision equivalent for each mesh and added the class to defaults.
    - Deleted `<body name="end_effector_link">` entirely and replaced with `<site name="pinch_site" pos="0 0 -0.061525" quat="0 1 0 0"/>` and added
    `<site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>` to defaults.
5. Added the following to the MJCF:
    - Added position actuators with joint limits.
    - Added keyframes for "home" and "retract". **(the specidied radians angles values should be adjusted)**
6. Deleted the unecessary <geom size ....> tags.
6. Added `scene.xml` which includes the robot, with a textured groundplane, skybox, and haze.

## Robotiq 2F-85

To be handled later

## License

This model is released under a [BSD-3-Clause License](LICENSE).

## Acknowledgement

This model was graciously contributed by [Jimmy Wu](https://jimmyyhwu.github.io/).
