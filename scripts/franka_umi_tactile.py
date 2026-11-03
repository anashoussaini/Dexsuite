"""Keyboard teleop for modular Franka with the UMI gripper (umi)."""

from __future__ import annotations

import argparse

import torch

import dexsuite as ds


def _make_env(task: str, render_mode: str | None, control_hz: int):
    control_dt = 1.0 / float(control_hz)
    return ds.make(
        task,
        sim=ds.options.SimOptions(control_hz=control_hz, n_envs=1),
        robot=ds.options.RobotOptions(
            single=ds.options.ArmOptions(
                manipulator="franka",
                gripper="umi",
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose",
                    config={
                        "normalized": False,
                        "clip": False,
                        "lin_scale": 0.2,
                        "ang_scale": 2.5,
                        "mink_dt": control_dt,
                        "mink_substeps": 8,
                        "max_joint_step": 0.08,
                        "hard_stop_on_zero": True,
                    },
                ),
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": False},
                ),
            ),
            visualize_tcp=False,
            visualize_aabb=False,
        ),
        render_mode=render_mode,
    )


def _pad_summary(name: str, pad_obs: dict[str, torch.Tensor]) -> str:
    contact = int(float(pad_obs["contact"].reshape(-1)[0].item()) > 0.5)
    force = pad_obs["force"].reshape(-1).detach().cpu()
    force_norm = float(pad_obs["force_norm"].reshape(-1)[0].item())
    fx, fy, fz = (float(v) for v in force[:3])
    return (
        f"{name}: contact={contact} "
        f"force=({fx:+.4f}, {fy:+.4f}, {fz:+.4f}) "
        f"|F|={force_norm:.4f}"
    )


class TactilePlotter:
    """Matplotlib viewer for the UMI tactile force grid."""

    def __init__(self) -> None:
        """Initialize the plot window and image handles."""
        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._rows = 12
        self._cols = 32
        self._pad_names = ("left_elastomer", "right_elastomer")
        self._fig, self._axes = plt.subplots(1, 2, figsize=(9, 4))
        self._images = [
            self._axes[0].imshow(
                torch.zeros((12, 32)).cpu().numpy(),
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            ),
            self._axes[1].imshow(
                torch.zeros((12, 32)).cpu().numpy(),
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            ),
        ]
        self._axes[0].set_title("left_elastomer")
        self._axes[1].set_title("right_elastomer")
        for ax in self._axes:
            ax.set_xticks([])
            ax.set_yticks([])
        self._fig.tight_layout()
        self._fig.show()

    def _grid_images(self, tactile_obs: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        points = tactile_obs["tactile_points_raw"].detach().cpu()
        if points.ndim == 3:
            points = points[0]

        num_sensors = len(self._pad_names)
        points_per_sensor = self._rows * self._cols
        points = points.view(num_sensors, points_per_sensor, 4).clone()

        normal_force = points[:, :, 3].clamp(min=0.0)
        max_values = normal_force.max(dim=-1).values
        low_mask = max_values < 0.0002
        if bool(low_mask.any().item()):
            normal_force[low_mask] /= 0.0008
        if bool((~low_mask).any().item()):
            normal_force[~low_mask] /= max_values[~low_mask].unsqueeze(-1)

        return normal_force.view(num_sensors, self._rows, self._cols)

    def update(self, tactile_obs: dict[str, dict[str, torch.Tensor]]) -> None:
        """Update the tactile heatmaps from the latest observation."""
        grids = self._grid_images(tactile_obs)
        for image, grid in zip(self._images, grids, strict=True):
            image.set_data(grid.numpy())
            image.set_clim(0.0, 1.0)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


def _print_controls() -> None:
    print("Keyboard teleop controls")
    print("  arrows : X / Y")
    print("  u / j  : Z up / down")
    print("  h / k  : roll")
    print("  y / g  : pitch")
    print("  b / n  : yaw")
    print("  o / p  : gripper open / close")
    print("  r      : reset env and zero target pose")
    print("  q / esc: quit")


def main() -> int:  # noqa: C901
    """Run the modular Franka tactile keyboard teleop loop."""
    parser = argparse.ArgumentParser(
        description="DexSuite modular Franka tactile keyboard teleop",
    )
    parser.add_argument(
        "--task",
        default="pick_place_pan",
        help="DexSuite task name to launch.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=5,
        help="Print tactile summaries every N environment steps.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open a matplotlib window with left/right tactile force maps.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("human", "none"),
        default="human",
        help="Viewer mode for Genesis.",
    )
    parser.add_argument(
        "--vis-mode",
        choices=("visual", "collision"),
        default="visual",
        help="Genesis surface visualization mode for the robot entities.",
    )
    parser.add_argument(
        "--lin-speed",
        type=float,
        default=0.10,
        help="Target translation speed in meters per second while a key is held.",
    )
    parser.add_argument(
        "--control-hz",
        type=int,
        default=20,
        help="Environment control rate; physics substeps remain managed by the task/env.",
    )
    parser.add_argument(
        "--ang-speed",
        type=float,
        default=1.0,
        help="Target rotation speed in radians per second while a key is held.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=2,
        help="Update the tactile plot every N steps to avoid UI-induced slowdown.",
    )
    args = parser.parse_args()
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.plot_every <= 0:
        raise ValueError("--plot-every must be > 0")

    from pynput import keyboard

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = _make_env(task=args.task, render_mode=render_mode, control_hz=args.control_hz)
    env.robot.arm_ent.surface.vis_mode = args.vis_mode
    env.robot.hand_ent.surface.vis_mode = args.vis_mode
    plotter = TactilePlotter() if args.plot else None
    device = env.device
    dt = float(getattr(env, "control_dt", 0.05))

    trans = {"x": 0.0, "y": 0.0, "z": 0.0}
    rot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    running = {"alive": True}
    reset_requested = {"value": False}

    grip_open = torch.as_tensor(
        env.robot.gripper.home_q,
        dtype=torch.float32,
        device=device,
    ).view(-1)
    grip_closed = torch.zeros_like(grip_open)
    grip_target = {"value": grip_open.clone()}
    key_axes = {
        keyboard.Key.up: (trans, "x", 1.0),
        keyboard.Key.down: (trans, "x", -1.0),
        keyboard.Key.left: (trans, "y", 1.0),
        keyboard.Key.right: (trans, "y", -1.0),
    }
    char_axes = {
        "u": (trans, "z", 1.0),
        "j": (trans, "z", -1.0),
        "h": (rot, "roll", 1.0),
        "k": (rot, "roll", -1.0),
        "y": (rot, "pitch", 1.0),
        "g": (rot, "pitch", -1.0),
        "b": (rot, "yaw", 1.0),
        "n": (rot, "yaw", -1.0),
    }

    def _set_axis(key, value: float) -> None:
        axis_spec = key_axes.get(key)
        if axis_spec is None:
            axis_spec = char_axes.get(str(getattr(key, "char", "")).lower())
        if axis_spec is None:
            return
        target, axis, sign = axis_spec
        target[axis] = sign * value

    def on_press(key):
        ch = getattr(key, "char", None)
        if key == keyboard.Key.esc or ch in ("q", "Q"):
            running["alive"] = False
            return False
        if ch in ("o", "O"):
            grip_target["value"] = grip_open.clone()
            return None
        if ch in ("p", "P"):
            grip_target["value"] = grip_closed.clone()
            return None
        if ch in ("r", "R"):
            reset_requested["value"] = True
            return None
        _set_axis(key, 1.0)
        return None

    def on_release(key):
        _set_axis(key, 0.0)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        obs, _ = env.reset()
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        total_steps = 0

        _print_controls()

        while running["alive"]:
            if reset_requested["value"]:
                obs, _ = env.reset()
                grip_target["value"] = grip_open.clone()
                reset_requested["value"] = False

            delta_pos = torch.tensor(
                [
                    dt * args.lin_speed * trans["x"],
                    dt * args.lin_speed * trans["y"],
                    dt * args.lin_speed * trans["z"],
                ],
                dtype=torch.float32,
                device=device,
            )
            delta_rpy = torch.tensor(
                [
                    dt * args.ang_speed * rot["roll"],
                    dt * args.ang_speed * rot["pitch"],
                    dt * args.ang_speed * rot["yaw"],
                ],
                dtype=torch.float32,
                device=device,
            )

            arm_cmd = torch.cat([delta_pos, delta_rpy], dim=0)
            action = torch.cat([arm_cmd, grip_target["value"]], dim=0)
            action = torch.minimum(torch.maximum(action, low), high)

            obs, reward, _terminated, _truncated, info = env.step(action)
            total_steps += 1

            tactile = obs["state"]["gripper"].get("tactile")
            if tactile is None:
                raise RuntimeError(
                    "gripper tactile observation is missing from obs['state']['gripper']",
                )

            if total_steps % args.print_every == 0:
                print(f"step={total_steps:05d} reward={float(reward.reshape(-1)[0]):+.4f}")
                print("  " + _pad_summary("left_elastomer", tactile["left_elastomer"]))
                print("  " + _pad_summary("right_elastomer", tactile["right_elastomer"]))

            if plotter is not None and total_steps % args.plot_every == 0:
                plotter.update(tactile)

            if bool(info["needs_reset"].reshape(-1)[0].item()):
                obs, _ = env.reset()
                grip_target["value"] = grip_open.clone()

        return 0
    finally:
        listener.stop()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
