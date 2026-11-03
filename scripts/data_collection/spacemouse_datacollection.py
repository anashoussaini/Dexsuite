"""Demonstrates controlling a single-arm LeRobot environment using a Spacemouse.

This script initializes a 'reach' task, captures user input from a Spacemouse
device, and passes the actions to the environment. It uses LeRobotLogger
to record and save only successful episodes.

Key Controls:
- Spacemouse: Controls the end-effector pose and gripper.
- Spacemouse Buttons (Reset): Pressing the reset button will drop the
  current trajectory and reset the environment without saving.

Episode Logic:
- Only episodes marked with 'success' are saved.
- Failed, terminated, or truncated episodes are dropped.
- A KeyboardInterrupt (Ctrl+C) will stop the script and finalize the
  dataset, discarding any in-progress (partial) episode.
"""

import torch

import dexsuite as ds
from dexsuite.devices import Spacemouse
from dexsuite.utils import get_device
from dexsuite.utils.lerobot_logger import LeRobotLogger


def main():
    """Main function to run the Spacemouse demonstration."""
    task = "lift"
    env = ds.make(
        task,
        manipulator="franka",
        gripper="allegro",
        arm_control="osc_pose",
        gripper_control="joint_position",
        render_mode="human",
        control_hz=20,
        cameras=["front", "wrist", "right_side"],
        n_envs=1,
    )
    logger = LeRobotLogger(task=task, env=env)
    obs, info = env.reset()
    dev = get_device()
    gripper_dim = int(env.robot.hand_ctrl.action_space().shape[0])

    sm = Spacemouse(
        path_or_ids=None,
        controller="OSCPose",
        normalized=True,
        deadzone=50,
    )

    try:
        try:
            while True:

                action = sm.get_action()
                action_list = action.get("pose") + action.get("gripper") * gripper_dim
                action_tensor = torch.tensor(
                    action_list,
                    dtype=torch.float32,
                    device=dev,
                )

                if action.get("reset") == [1] or action.get("reset") is True:
                    logger.clear_buffer()
                    obs, info = env.reset()
                    continue

                # 2) take the step
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
                    print("SUCCESS - saving episode")
                    logger.save_episode()
                    obs = next_obs
                elif is_done:
                    print("END (failure/terminated/truncated) - dropping attempt")
                    logger.clear_buffer()
                    obs = next_obs
                else:
                    # continue rollout
                    obs = next_obs
        except KeyboardInterrupt:
            print("\nInterrupted by user. Not saving partial attempts (by design).")

        logger.close()
        print("Dataset finalized.")

    finally:
        env.close()
        sm.close()
        print("Environment and Spacemouse closed.")


if __name__ == "__main__":
    main()
