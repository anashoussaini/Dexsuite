"""Manual single-arm environment smoke test with Vive/Manus inputs."""

import argparse

import numpy as np
import torch

import dexsuite as ds
from dexsuite.devices import ZMQViveTracker
from dexsuite.devices.external_reset_helper import ExternalResetHelper
from dexsuite.devices.manus_geort import ManusGeort

ROBOTS = {
    "grip": ["franka", "robotiq"],
    "integrated": ["trossen_vx300s"],
    "hand": ["franka", "allegro_left"],
}


def get_device() -> torch.device:
    """Determines and returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected torch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(task, manip, gripper, d, e):  # noqa: C901
    """Run an interactive single-arm environment smoke test."""
    n_envs = 4 if e else 1
    # manip = "franka_with_gripper"
    try:
        vt = ZMQViveTracker(endpoint="tcp://127.0.0.1:5559", topic="vive.tracker")
    except Exception as ex:
        print("ZMQ Vive Tracker could not be loaded")
        print(ex)
        vt = None

    env = ds.make(
        task,
        sim=ds.options.SimOptions(control_hz=20, n_envs=n_envs),
        robot=ds.options.RobotOptions(
            visualize_aabb=True,
            single=ds.options.ArmOptions(
                manipulator=manip,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                    config={"normalized": False},
                ),
                gripper=gripper,
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": False},
                )

            ),
            
        ),
        render_mode="human",
        cameras=["wrist"],
        camera_gui=True,
    )
    gripper_dim = int(env.robot.hand_ctrl.action_space().shape[0])

    manus = ManusGeort(url="ws://127.0.0.1:8767", expected_dim=gripper_dim)
    trigger_keys = {"r": None, "d": env.reset}
    if vt is not None:
        trigger_keys = {"r": vt.recalibrate, "d": env.reset}
    ext_reset = ExternalResetHelper(trigger_keys)
    dev = get_device()
    env.reset()
    print(env.robot._layout)
    print("Task:", task)
    print("Manipulator:", manip)
    print("Gripper:", gripper)

    counter = 0
    action = {"pose": [0, 0, 0, 1, 0, 0, 0], "gripper": [0]}
    try:
        try:
            while True:
                counter += 1
                if vt is not None:
                    action = vt.get_action()
                x, y, z, qw, qx, qy, qz = action.get("pose")
                grip = manus.get_action()
                action_list = np.concatenate(
                    [np.array([x, y, z, qw, qx, qy, qz]), grip],
                )
                if e:
                    action_list = [action_list, action_list, action_list, action_list]

                action_tensor = torch.tensor(
                    action_list,
                    dtype=torch.float32,
                    device=dev,
                )
                with torch.no_grad():
                    _next_obs, _reward, terminated, truncated, info = env.step(
                        action_tensor,
                    )

                success = info.get("success", [False for _ in range(n_envs)]).to(
                    torch.bool,
                )

                is_done = torch.logical_or(
                    terminated.to(torch.bool),
                    truncated.to(torch.bool),
                )
                if ext_reset.key_pressed() == "d":
                    is_done = True
                if e:
                    is_done = [i + 1 <= counter % 4 for i in range(4)]
                    for i in range(n_envs):
                        if success[i]:
                            print(f"SUCCESS for env {i}")
                            env.reset_env_idx([i])
                        elif is_done[i]:
                            print(f"END (failure/terminated/truncated)  for env {i}")

                            env.reset_env_idx([i])
                elif success:
                    print("SUCCESS ")
                    env.reset()
                elif is_done:
                    print("END (failure/terminated/truncated) ")

                    env.reset()
                if d:
                    print("RESETING DUE TO SETTING d")
                    env.reset()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script for testing environments")
    parser.add_argument("arg1", help="<a,b,c>: the type of robot")
    parser.add_argument("arg2", help="<d,e>: reset or other options")
    parser.add_argument(
        "--task",
        type=str,
        default="lift",
        help="Specify the task name.",
    )

    args = parser.parse_args()

    task = args.task

    manip = ROBOTS[args.arg1][0]

    gripper = ROBOTS[args.arg1][1] if len(ROBOTS[args.arg1]) > 1 else None

    flag_d = args.arg2 == "d"
    flag_e = args.arg2 == "e"
    main(task, manip, gripper, flag_d, flag_e)
