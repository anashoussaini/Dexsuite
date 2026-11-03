# Arm-to-gripper adapter YAML

DexSuite uses adapter YAML files to describe how a gripper is mounted to a
modular manipulator. Adapters are only used for modular manipulators (where the
gripper is a separate model). Integrated manipulators include their gripper in
the main MJCF and do not use adapter catalogs.

Where adapters are used
-----------------------

- `dexsuite.core.components.arm.ModularManipulatorModel` exposes an `adapters`
  catalog loaded from YAML.
- `dexsuite.core.robots.modular_robot.ModularRobot` looks up an adapter entry
  for the chosen gripper and constructs `dexsuite.core.components.mount.GripperMount`.

How the YAML file is located
----------------------------

The catalog is loaded by `dexsuite.core.components.arm.ArmModel._load_adapter_yaml()`
using this search order:

1. The explicit `_adapter_yaml` path set on the arm class (absolute, or relative
   to the arm's Python file).
2. `<registry_name>_adapters.yaml` next to the arm's Python file.
3. `<module_stem>_adapters.yaml` next to the arm's Python file.

If no file is found, the catalog is empty.

Schema
------

The YAML is a mapping from a gripper registry key to a dict of keyword
arguments for `GripperMount`:

- `mount_type`: Adapter geometry type (default: "invisible").
- `mount_pos`: [x, y, z] position of the mount relative to the arm flange.
- `mount_quat`: [x, y, z, w] orientation of the mount relative to the arm flange.
- `geometry`: Optional geometry description when a visible mount is used.
- `gripper_pos`: [x, y, z] position of the gripper root relative to the mount.
- `gripper_quat`: [x, y, z, w] orientation of the gripper root relative to the mount.

Example:

```yaml
robotiq:
  mount_type: invisible
  mount_pos: [0.0, 0.0, 0.0]
  mount_quat: [0.0, 0.0, 0.0, 1.0]
  gripper_pos: [0.0, 0.0, 0.005]
  gripper_quat: [0.0, 0.0, 0.0, 1.0]
```

Troubleshooting
---------------

If you see an error similar to "Adapter missing ... Available: [...]":

- Add an entry for the gripper key to the arm's adapter YAML file, or
- Select a gripper key that exists in the adapter catalog.
| **Add a new arm**       | 1) Create `myarm.py` subclassing `ArmModel`.<br>2) Copy `franka_adapters.yaml` → `myarm_adapters.yaml`, tune offsets per gripper. |
| **Override at runtime** | Provide a custom YAML path:<br>`FrankaArm._adapter_yaml = "/tmp/test_mounts.yaml"` before instantiating the env.                  |

---

### 6. Debugging tips

```python
print(env.robot.get_mount_info())
# → {'visible': True, 'type': 'cylinder', 'pos_offset': [...], ...}
```

Run `dexsuite/tests/mount_demo.py` for a live preview with viewer.

---

Happy mounting!
