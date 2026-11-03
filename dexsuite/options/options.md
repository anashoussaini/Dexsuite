# DexSuite API Guide

This document provides an overview of the DexSuite environment construction API
and the options objects in `dexsuite.options`. Examples are intended to be
minimal and copy/paste friendly.

***

## 1. The Core Function: `ds.make`

The `ds.make` function is how you build a DexSuite environment. It takes the environment name (e.g., `"stack"`, `"bimanual_reach"`) and configuration options.

There are two equally supported styles for passing configuration:

### A. Options Object API

You pass explicit, strongly-typed option objects (`SimOptions`, `RobotOptions`, etc.). This approach is generally cleaner and loads sensible defaults from your YAML configuration.

```python
import dexsuite as ds
from dexsuite.options import RobotOptions, ArmOptions, CamerasOptions, SimOptions

env = ds.make(
    "stack",
    sim=SimOptions(control_hz=100),
    robot=RobotOptions(  # Configures a single arm by default
        single=ArmOptions(manipulator="franka", gripper="robotiq")
    ),
    cameras=CamerasOptions(modalities=("rgb",)),  # Simple camera pack
    render_mode="rgb_array",
)
```

### B. Direct API

You pass strings and tuples directly as top-level arguments, and DexSuite automatically builds the necessary options objects internally.

```python
# Single Arm
env = ds.make(
    "stack",
    manipulator="franka",
    gripper="robotiq",
    arm_control="osc_pose",
    gripper_control="joint_position",
    cameras=["front", "wrist"],    # Uses named presets from YAML
    modalities=("rgb",),
    render_mode="rgb_array",
)

# Bimanual Arms (Arms defined as tuples)
env = ds.make(
    "bimanual_reach",
    manipulator=("franka", "franka"),
    gripper=("robotiq", "robotiq"),
    layout="side_by_side",         # Preset name (bimanual only)
    cameras=["front", "wrist"],    # "wrist" expands to left/right wrists
)
```

Exclusivity rule: Do not mix the `robot=RobotOptions(...)` argument with the
direct robot configuration keys (like `manipulator`, `gripper`, `arm_control`,
etc.) in the same `ds.make` call.

### Accepted `ds.make` keyword arguments (strict)

DexSuite raises `TypeError` on unknown `ds.make(...)` keywords (no silent ignores).

**Options Object API**
- `robot`: `RobotOptions | None`
- `sim`: `SimOptions | None`
- `cameras`: `CamerasOptions | Sequence[str] | None`
- `layout`: `LayoutOptions | str | None` (string presets are bimanual-only)
- `render_mode`: `"human" | "rgb_array" | None`

**Direct API**
- Robot: `manipulator`, `gripper`, `arm_control`, `gripper_control`
- Cameras: `cameras`, `modalities`
- Sim: `control_hz`, `performance_mode`, `n_envs` (alias: `n_env`)

-----

## 2\. Components at a Glance

DexSuite uses four main options blocks. You can pass any subset, and the rest will be filled from YAML defaults.

| Option Block | Description | Key Fields |
| :--- | :--- | :--- |
| **`SimOptions`** | Simulation timing and flags. | `control_hz: int`, `performance_mode: bool` |
| **`RobotOptions`** | Hardware, controllers, and arm layouts. | `type_of_robot`, `single`/`left`/`right` (`ArmOptions`), `layout` (`LayoutOptions`) |
| **`ArmOptions`** | Specs **per arm**. | `manipulator: str`, `gripper: Optional[str]`, `manipulator_controller`, `gripper_controller`, `workspace` |
| **`CamerasOptions`** | Static, dynamic cameras, and observation modalities. | `static: dict`, `dynamic: dict`, `modalities: tuple[str, ...]` |

-----

## 3\. Single vs. Bimanual Robots

The robot structure changes based on the number of arms:

  * **Single Arm:** Uses one `ArmOptions` object under `robot.**single**`. It **does not** use layout presets. For non-origin placement, set `robot.layout.single = PoseOptions(...)`.
    ```python
    robot=RobotOptions(single=ArmOptions(manipulator="franka", ...))
    ```
  * **Bimanual Arms:** Requires `robot.**left**` and `robot.**right**` and a **layout preset** (e.g., `"side_by_side"`) to determine the base poses. If omitted, it defaults to `"side_by_side"`.
    ```python
    env = ds.make(
        "bimanual_reach",
        manipulator=("franka", "franka"),
        layout="right_angle",  # Bimanual requires a layout preset
        # ... other options
    )
    ```

> **Layout Preset Resolution:** Presets are resolved **per manipulator model** using `config/layouts/<model>.yaml`, falling back to `config/layouts/defaults.yaml`.

-----

## 4\. Integrated vs. Modular Arms

Arms are classified based on how the gripper is attached:

  * **Integrated Arms:** The manipulator model includes the hand/gripper (e.g., `"franka_with_gripper"`).
      * Set `gripper=None` (or omit it).
      * You still configure a `gripper_controller` if the integrated hand is controllable.
  * **Modular Arms:** The manipulator and gripper are separate.
      * You **must** specify a `gripper` name that the manipulator can mount.
    <!-- end list -->
    ```python
    # Modular example
    env = ds.make(
        "stack",
        manipulator="franka",      # Manipulator only
        gripper="robotiq",         # Required for modular arms
        gripper_control="joint_position"
    )
    ```

> **Controller Normalization:** Controllers default to **normalized** inputs (actions are $\in [-1, 1]$). To use physical units, pass `ControllerOptions(config={"normalized": False})` to the appropriate controller block.

-----

## 5\. Workspaces (AABB)

The operational workspace is defined by an **Axis-Aligned Bounding Box (AABB)** in `config/workspaces.yaml`.

  * The key in the YAML must **exactly match** the manipulator name (e.g., `"franka"`, `"franka_with_gripper"`). Integrated variants must be listed separately.
  * DexSuite strictly enforces this: if a manipulator lacks a workspace entry, it raises an error.

<!-- end list -->

```yaml
# config/workspaces.yaml excerpt
workspaces:
  franka:
    min: [0.10, -0.50, 0.00]
    max: [0.80,  0.50, 0.70]
  franka_with_gripper: # Separate entry required
    min: [0.20, -0.50, 0.00]
    max: [0.55,  0.50, 0.65]
```

The **world AABB** is computed by translating the arm's AABB (defined in the arm's base frame) by the base position from the layout. Yaw rotation is ignored; the box remains axis-aligned in the world frame. For bimanual setups, the world AABB is the union of the left and right arm boxes.

-----

## 6\. Layouts & Poses

### Layout Presets (Bimanual Only)

Bimanual setups use layout presets (strings like `"side_by_side"` or `"right_angle"`) to determine left and right arm base poses.

### Explicit Pose Overrides

You can explicitly override base poses using `PoseOptions`:

  * **Single Arm:** Use `robot.layout.single`.
  * **Bimanual Arms:** You can pass a preset **and** override one or both sides.

<!-- end list -->

```python
from dexsuite.options import LayoutOptions, PoseOptions, RobotOptions, ArmOptions

rob = RobotOptions(
    type_of_robot="bimanual",
    left=ArmOptions(manipulator="franka", gripper="robotiq"),
    right=ArmOptions(manipulator="franka", gripper="robotiq"),
    layout=LayoutOptions(
        preset="side_by_side",
        # Override the left arm's position
        left=PoseOptions(pos=(0.00, 0.45, 0.00), yaw_rad=+1.5708)
    ),
)
env = ds.make("bimanual_reach", robot=rob)
```

-----

## 7\. Cameras: Static, Dynamic, Modalities

### Quick Presets (Direct API)

Use a list of camera names (looked up in `config/cameras.yaml`).

```python
env = ds.make("stack", cameras=["front", "wrist"], modalities=("rgb", "depth"))
```

  * `"front"`: A static camera preset.
  * `"wrist"`: A special dynamic camera. For bimanual, it auto-expands to `"left_wrist"` and `"right_wrist"`.

### Full Control: `CamerasOptions`

For fine-grained control, use `CamerasOptions` for static (fixed position) and dynamic (attached to a robot link) cameras.

Bimanual dynamic naming: For bimanual setups, dynamic camera names must be
prefixed with `"left_"` or `"right_"` (for example, `"left_wrist"`). The prefix
determines which arm the camera is attached to.

```python
from dexsuite.options import CamerasOptions, DynamicCamOptions

cams = CamerasOptions(
    static={...},
    dynamic={
        "left_wrist": DynamicCamOptions(
            pos_offset=(0.0, 0.0, 0.05), # offset in link frame
            # ... other options
        ),
        "right_wrist": "wrist_cam",    # Use a preset for the right wrist
    },
    modalities=("rgb",),
)
env = ds.make("bimanual_reach", cameras=cams, manipulator=("franka","franka"))
```

-----

## 8\. Actions & Observations

DexSuite maintains a consistent structure for control and data.

### Action Spaces

Actions are normalized by default to values in [-1, 1].

| Robot Type | Action Space Structure |
| :--- | :--- |
| **Single (Modular/Integrated)** | `Box(shape=(D,), low=-1, high=1)` |
| **Bimanual** | `Box(shape=(D,), low=-1, high=1)` |

DexSuite accepts **only `torch.Tensor` actions** and uses a *flat* action vector
internally. You can inspect `env.robot._layout` (segment names + dims) if you
need to slice actions by controller.

```python
import torch

# Single env: action shape is (D,)
act = torch.zeros(env.action_space.shape, dtype=torch.float32)
obs, reward, terminated, truncated, info = env.step(act)
```

### Observation Structure

Observations are organized into `"state"` (robot and environment data) and `"cameras"` (visual data).

```python
obs = {
  "state": {
    "manipulator": { "qpos", "eef_pose", ... }, # Single
    "gripper": { "tcp_pose", "grasp": {...}, ... },
    # Bimanual has "left" and "right" instead of "manipulator" and "gripper"
    # "other": {...} # Optional environment extras
  },
  "cameras": {
    "front": { "rgb": (H,W,3), "depth": (H,W), ... },
    "left_wrist": { "rgb": ... },
    ...
  }
}
```

-----

## 9\. Controllers - Names and rules

Controllers define how actions map to the simulation.

| Component | Allowed Controllers | Notes |
| :--- | :--- | :--- |
| **Manipulator** | `"joint_position"`, `"joint_velocity"`, `"joint_torque"`, `"osc_pose"`, `"osc_pose_abs"`, `"osc_pose_abs_quat"`, `"ik_pose"` | Any of these. Pose controllers default to `backend="mink"` and can use Genesis IK with `config={"backend": "genesis_internal"}`. |
| **Gripper** | `"joint_position"`, `"joint_velocity"`, `"joint_torque"` | **Only** joint-level controllers are allowed. Using OSC/IK will raise an error. |

-----

## 10\. Multi-Environment & Stepping

  * **Multi-Env:** Pass `n_envs=<N>` to `ds.make(...)` (alias: `n_env`) to create vectorized environments on a single scene. Actions/observations will be batched.
  * **Stepping:** The default simulation horizon is 200 steps. Always call `obs, _ = env.reset()` before stepping.

-----

## 11\. Common Recipes

| Scenario | Options Object API | Direct API |
| :--- | :--- | :--- |
| **Single, Defaults** | `env = ds.make("stack")` | `env = ds.make("stack")` |
| **Bimanual, Preset** | `robot=RobotOptions(type_of_robot="bimanual", ...)` | `env = ds.make("bimanual_reach", manipulator=("f","f"), gripper=("r","r"), layout="side_by_side", cameras=["wrist"])` |
| **Integrated Arm** | `robot=RobotOptions(single=ArmOptions(manipulator="franka_with_gripper", gripper=None))` | `env = ds.make("stack", manipulator="franka_with_gripper", gripper=None)` |

-----

## 12\. Troubleshooting & Gotchas

  * **"Dynamic camera 'left\_wrist' did not match any arm"**: You're in a bimanual setup, but the dynamic camera name is missing the `left_` or `right_` prefix in `CamerasOptions`. Use `cameras=["wrist"]` in the direct API for auto-expansion.
  * **"Single-arm robots do not use layout presets"**: Presets (strings like `"side_by_side"`) are bimanual-only. For single-arm placement, use `robot.layout.single = PoseOptions(...)`.
  * **"No workspace AABB specified for manipulator 'X'"**: Add an exact-match entry to `config/workspaces.yaml`. Integrated arms need their own entry (e.g., `"franka_with_gripper"`).
  * **Gripper Controller Errors**: Only use `joint_*` controllers for grippers.
  * **Mixing `robot=` with direct fields**: Use one style or the other in `ds.make`.

-----

## 13\. Mental Model Recap

  * **Configuration Split:** Think of **`RobotOptions`** as "**what's in the scene**" (hardware + controllers), and **`LayoutOptions`** as "**where you put it**" (base pose).
  * **Workspaces:** Strict, single source (`workspaces.yaml`), keyed by exact manipulator names.
  * **Bimanual:** Uses string presets for sensible base placement, but you can override with explicit poses.
  * **Cameras:** `static` (fixed world position) vs. `dynamic` (attached to robot links). Dynamic names for bimanual robots must be prefixed to attach to the correct arm.
