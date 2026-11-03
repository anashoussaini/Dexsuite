"""Collect LeRobot episodes with Vive Tracker arm control and Manus hand input.

This script runs a DexSuite environment, uses a Vive Tracker to command the arm
pose, and reads a Manus glove joint stream (via WebSocket) to command the hand
gripper joints. Successful episodes are recorded with LeRobotLogger.
"""

import numpy as np
import torch

import dexsuite as ds
from dexsuite.devices.external_reset_helper import ExternalResetHelper
from dexsuite.devices.manus_geort import ManusGeort
from dexsuite.devices.vive_tracker import ViveTracker
from dexsuite.utils.lerobot_logger import LeRobotLogger

CTRL_HZ = 20


def get_device() -> torch.device:
    """Return the torch device used for actions.

    Returns:
        The preferred torch device (CUDA if available, otherwise CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_bool(val) -> bool:
    """Converts a potential tensor, array, or scalar value to a simple boolean.

    Args:
        val: The value to convert. Can be a torch.Tensor, np.ndarray,
             bool, or int.

    Returns:
        bool: The Python boolean representation of the value.
    """
    if hasattr(val, "item"):
        return bool(val.item())
    return bool(val)





def main():
    """Run Manus-based hand data collection."""
    task = "bimanual_stack_mid_air"
    manip = "ur5"
    gripper = "allegro"
    env = ds.make(
        task,
        sim=ds.options.SimOptions(control_hz=CTRL_HZ),
        robot=ds.options.RobotOptions(
            visualize_aabb=False,
            type_of_robot="bimanual",
            left=ds.options.ArmOptions(
                manipulator=manip,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                    config={"normalized": False},
                ),
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": False},
                ),
                gripper=gripper+"_left",
            ),
            right=ds.options.ArmOptions(
                manipulator=manip,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                    config={"normalized": False},
                ),
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": False},
                ),
                gripper=gripper+"_right",
            ),
        ),
        cameras=["front", "wrist"],
        render_mode="human",
    )
    gripper_dim = int(env.robot.left.hand_ctrl.action_space().shape[0])

    # Tracker input
    vt_right = ViveTracker(tracker_id="tracker_1", side="right")
    manus_right = ManusGeort(url="ws://127.0.0.1:8767", expected_dim=gripper_dim)

    vt_left = ViveTracker(tracker_id="tracker_2", side="left")
    manus_left = ManusGeort(url="ws://127.0.0.1:8768", expected_dim=gripper_dim)
    logger = LeRobotLogger(task=task, env=env)

    trigger_keys = {"r": vt_right.recalibrate, "l": vt_left.recalibrate, "d": None}
    ext_reset = ExternalResetHelper(trigger_keys)


    obs, info = env.reset()
    dev = get_device()
    count = 0
    try:
        try:
            while True:
                if ext_reset.key_pressed() == "d":
                    logger.clear_buffer()
                    obs, info = env.reset()
                    continue

                action_left = vt_left.get_action()
                action_right = vt_right.get_action()
                grip_left = manus_left.get_action()
                

                grip_right = manus_right.get_action()
                
                action_left = np.concatenate([action_left.get("pose"), grip_left])
                action_right = np.concatenate([action_right.get("pose"), grip_right])
                action_list = np.concatenate([action_left, action_right])
                action_tensor = torch.tensor(
                    action_list,
                    dtype=torch.float32,
                    device=dev,
                )

                with torch.no_grad():
                    next_obs, reward, terminated, truncated, info = env.step(
                        action_tensor,
                    )
                    
                logger.add_step_pre(
                    obs_before=obs,
                    action=action_tensor,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )

                success = bool(info.get("success", False))

                is_done = bool(terminated) or bool(truncated)

                if success:
                    print(logger.ds.episode_buffer["size"])
                    count += 1
                    print(f"SUCCESS — saving episode {count}")

                    logger.save_episode()

                    print(f"SUCCESS — saved episode {count}")
                    wait_for_reset(ext_reset, logger)
                    obs = next_obs

                elif is_done:
                    print("END (failure/terminated/truncated) — dropping attempt")
                    logger.clear_buffer()
                    wait_for_reset(ext_reset, logger)
                    obs = next_obs

                else:
                    obs = next_obs
        except KeyboardInterrupt:
            print("\nInterrupted by user. Not saving partial attempts (by design).")

        logger.close()
        print("Dataset finalized.")

    finally:
        ext_reset.close()
        env.close()
        print("Environment and ViveController closed.")


def wait_for_reset(reset_helper, logger):
    """Wait for right and left tracker recalibration before the next episode."""
    reset_right = False
    reset_left = False
    print(
        "Episode: Please go back to origin position and recalibrate "
        "(right first, then left).",
    )
    while not reset_right:
        out = reset_helper.key_pressed()
        if out == "r":


            reset_right = True
    while not reset_left:
        out = reset_helper.key_pressed()
        if out == "l":

            reset_left = True
    logger.clear_buffer()

if __name__ == "__main__":
    main()
