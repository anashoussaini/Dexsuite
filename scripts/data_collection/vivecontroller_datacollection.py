"""Collect LeRobot episodes with a Vive Controller.

This script runs a DexSuite environment, reads pose and gripper commands from a
Vive Controller, and logs successful episodes using LeRobotLogger.

Controls:
  - Trackpad press: reset the environment and drop the current buffer.

Episode policy:
  - Save episodes only when the environment reports success.
  - Drop episodes on termination or truncation.
"""

import argparse
import contextlib
from datetime import datetime
from pathlib import Path

import torch

import dexsuite as ds
from dexsuite.devices.vive_controller import (
    DEFAULT_ZMQ_ENDPOINT,
    DEFAULT_ZMQ_TOPIC,
    ViveController,
)
from dexsuite.utils.lerobot_logger import LeRobotLogger
try:
    from scripts.data_collection.autopauser import AutoPauser
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from autopauser import AutoPauser

CTRL_HZ = 20
DEFAULT_TASK = "pick_place_fruit"
DEFAULT_MANIPULATOR = "ur5"
DEFAULT_END_EFFECTOR = "robotiq"
DEFAULT_DATASET_ROOT = "collected_dataset"
DEFAULT_CONTROLLER_ID = "controller_1"
DEFAULT_CONTROLLER_SIDE = "left"
DEFAULT_CONTROLLER_SOURCE = "auto"
DEFAULT_CONTROLLER_TRANSPORT = "zmq"
DEFAULT_CONTROLLER_ENDPOINT = DEFAULT_ZMQ_ENDPOINT
DEFAULT_CONTROLLER_TOPIC = DEFAULT_ZMQ_TOPIC
DEFAULT_CONTROLLER_DISCOVERY_TIMEOUT_SEC = 3.0
DEFAULT_CONTROLLER_RECV_TIMEOUT_MS = 100
DEFAULT_CONTROLLER_STALE_TIMEOUT_SEC = 0.5
DEFAULT_CONTROLLER_HAPTICS = True
DEFAULT_AUTO_PAUSE_THRESHOLD = 0.005
DEFAULT_GRIP_HOLD_SEC = 0.5
VIVE_CONTROLLER_ACTION_DIM = 8


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


def _safe_path_part(value: str) -> str:
    """Sanitize a value so it can be used as one component of a path.

    Args:
        value: Raw value to include in a generated dataset path.

    Returns:
        A non-empty string containing only alphanumeric characters, dashes, and
        underscores.
    """
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe.strip("_") or "unnamed"


def _collection_dataset_dir(
    dataset_root: str | Path,
    task: str,
    manipulator: str,
    end_effector: str,
) -> Path:
    """Build the timestamped output directory for a collection run.

    Args:
        dataset_root: Root directory under which datasets are stored.
        task: DexSuite task name.
        manipulator: Manipulator model name.
        end_effector: End-effector or gripper model name.

    Returns:
        Path to the run-specific dataset directory.
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    task_name = _safe_path_part(task)
    manip_name = _safe_path_part(manipulator)
    ee_name = _safe_path_part(end_effector)
    dataset_name = f"{task_name}_{manip_name}_{ee_name}_{timestamp}"
    return Path(dataset_root) / task_name / dataset_name


def main(
    *,
    task: str = DEFAULT_TASK,
    manipulator: str = DEFAULT_MANIPULATOR,
    end_effector: str = DEFAULT_END_EFFECTOR,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    controller_id: str = DEFAULT_CONTROLLER_ID,
    controller_side: str = DEFAULT_CONTROLLER_SIDE,
    controller_source: str = DEFAULT_CONTROLLER_SOURCE,
    controller_transport: str = DEFAULT_CONTROLLER_TRANSPORT,
    controller_endpoint: str = DEFAULT_CONTROLLER_ENDPOINT,
    controller_topic: str = DEFAULT_CONTROLLER_TOPIC,
    controller_discovery_timeout_sec: float = DEFAULT_CONTROLLER_DISCOVERY_TIMEOUT_SEC,
    controller_recv_timeout_ms: int = DEFAULT_CONTROLLER_RECV_TIMEOUT_MS,
    controller_stale_timeout_sec: float = DEFAULT_CONTROLLER_STALE_TIMEOUT_SEC,
    haptics: bool = DEFAULT_CONTROLLER_HAPTICS,
    auto_pause_threshold: float = DEFAULT_AUTO_PAUSE_THRESHOLD,
    grip_hold_sec: float = DEFAULT_GRIP_HOLD_SEC,
):
    """Run Vive Controller data collection.

    Args:
        task: DexSuite task name to instantiate.
        manipulator: Manipulator model used by the single-arm robot.
        end_effector: End-effector or gripper model used by the robot.
        dataset_root: Root directory for saved LeRobot datasets.
        controller_id: Vive controller name or serial to select.
        controller_side: Controller role label to select, such as left or right.
        controller_source: Pose representation source used by the controller.
        controller_transport: Backend transport, such as zmq, openvr, or survive.
        controller_endpoint: ZMQ endpoint used when the transport is zmq.
        controller_topic: ZMQ topic used when the transport is zmq.
        controller_discovery_timeout_sec: Seconds to wait for initial samples.
        controller_recv_timeout_ms: Per-receive ZMQ timeout in milliseconds.
        controller_stale_timeout_sec: Maximum age for fresh controller samples.
        haptics: Whether controller haptic feedback is enabled.
        auto_pause_threshold: Per-action-dimension movement threshold for logging.
        grip_hold_sec: Seconds to continue stepping after a gripper close toggle.
    """
    env = ds.make(
        task,
        sim=ds.options.SimOptions(control_hz=CTRL_HZ),
        robot=ds.options.RobotOptions(
            single=ds.options.ArmOptions(
                manipulator=manipulator,
                gripper=end_effector,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                    config={"normalized": False},
                ),
            ),
        ),
        render_mode="human",
    )
    dataset_dir = _collection_dataset_dir(
        dataset_root,
        task,
        manipulator,
        end_effector,
    )
    print(f"Saving collected dataset to {dataset_dir}", flush=True)
    logger = LeRobotLogger(task=task, env=env, dataset_dir=dataset_dir)
    autopauser = AutoPauser(
        VIVE_CONTROLLER_ACTION_DIM,
        [auto_pause_threshold] * VIVE_CONTROLLER_ACTION_DIM,
    )
    obs, info = env.reset()
    dev = get_device()

    print(
        "Connecting Vive Controller with "
        f"transport={controller_transport!r} endpoint={controller_endpoint!r} "
        f"topic={controller_topic!r} controller_id={controller_id!r} "
        f"side={controller_side!r}.",
        flush=True,
    )
    vc = ViveController(
        controller_id=controller_id,
        side=controller_side,
        source=controller_source,
        transport=controller_transport,
        endpoint=controller_endpoint,
        topic=controller_topic,
        discovery_timeout_sec=controller_discovery_timeout_sec,
        recv_timeout_ms=controller_recv_timeout_ms,
        stale_timeout_sec=controller_stale_timeout_sec,
        haptics=haptics,
    )
    grip = False
    grip_countdown = 0
    count = 0
    try:
        try:
            while True:
                action = vc.get_action()
                x, y, z, qw, qx, qy, qz = action.get("pose")
                gripper = float(action.get("gripper")[0])
                if gripper == 1:
                    grip = not grip
                    grip_countdown = int(grip_hold_sec * CTRL_HZ)
                g = -1 if grip else 1
                action_list = [x, y, z, qw, qx, qy, qz, g]

                if action.get("buttons").get("trackpad_pressed"):
                    logger.clear_buffer()
                    print("[RESET] Environment Reset")
                    obs, info = env.reset()
                    grip = True
                    continue
                if not autopauser.check_movement(action_list):
                    if grip_countdown > 0:
                        grip_countdown -= 1
                    else:
                        continue

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
                    print(f"SUCCESS {count} - saving episode")

                    logger.save_episode()
                    print(f"SUCCESS {count}— episode saved")
                    grip = True
                    wait_for_reset(vc, logger)
                    obs = next_obs

                elif is_done:
                    print("END (failure/terminated/truncated) - dropping attempt")
                    logger.clear_buffer()
                    grip = True
                    wait_for_reset(vc, logger)
                    obs = next_obs

                else:
                    obs = next_obs
        except KeyboardInterrupt:
            print("\nInterrupted by user. Not saving partial attempts (by design).")

        logger.close()
        print("Dataset finalized.")

    finally:
        with contextlib.suppress(Exception):
            vc.close()
        env.close()
        print("Environment and ViveController closed.")


def wait_for_reset(vc, logger):
    """Block until the user confirms the next episode can start.

    Args:
        vc: Vive Controller instance.
        logger: LeRobotLogger used to clear any buffered steps.
    """
    print("Please go back to origin position and recalibrate")
    while True:
        action = vc.get_action()
        if action.get("buttons").get("grip_button", False) or action.get("buttons").get(
            "grip",
            False,
        ):
            logger.clear_buffer()
            return


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Vive Controller data collection.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect single-arm episodes with a Vive Controller.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="DexSuite task name.",
    )
    parser.add_argument(
        "--manipulator",
        type=str,
        default=DEFAULT_MANIPULATOR,
        help="Manipulator model name to include in the env and dataset path.",
    )
    parser.add_argument(
        "--end-effector",
        type=str,
        default=DEFAULT_END_EFFECTOR,
        help="End-effector/gripper model name to include in the env and dataset path.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory for collected datasets.",
    )
    parser.add_argument(
        "--controller-id",
        type=str,
        default=DEFAULT_CONTROLLER_ID,
        help="Remote Vive controller name or serial.",
    )
    parser.add_argument(
        "--controller-side",
        type=str,
        default=DEFAULT_CONTROLLER_SIDE,
        help="Controller role label to select, such as left, right, or any.",
    )
    parser.add_argument(
        "--controller-source",
        type=str,
        default=DEFAULT_CONTROLLER_SOURCE,
        help="Controller pose source: auto, quat, euler/rpy, or matrix.",
    )
    parser.add_argument(
        "--controller-transport",
        type=str,
        default=DEFAULT_CONTROLLER_TRANSPORT,
        help="Controller backend transport: zmq, openvr, or survive.",
    )
    parser.add_argument(
        "--controller-endpoint",
        type=str,
        default=DEFAULT_CONTROLLER_ENDPOINT,
        help="Vive controller ZMQ endpoint to connect to.",
    )
    parser.add_argument(
        "--controller-topic",
        type=str,
        default=DEFAULT_CONTROLLER_TOPIC,
        help="Vive controller ZMQ topic to subscribe to.",
    )
    parser.add_argument(
        "--controller-discovery-timeout-sec",
        type=float,
        default=DEFAULT_CONTROLLER_DISCOVERY_TIMEOUT_SEC,
        help="Seconds to wait for initial Vive controller samples.",
    )
    parser.add_argument(
        "--controller-recv-timeout-ms",
        type=int,
        default=DEFAULT_CONTROLLER_RECV_TIMEOUT_MS,
        help="Per-receive ZMQ timeout in milliseconds.",
    )
    parser.add_argument(
        "--controller-stale-timeout-sec",
        type=float,
        default=DEFAULT_CONTROLLER_STALE_TIMEOUT_SEC,
        help="Seconds before the latest Vive controller sample is stale.",
    )
    parser.add_argument(
        "--haptics",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONTROLLER_HAPTICS,
        help="Enable controller haptic feedback.",
    )
    parser.add_argument(
        "--auto-pause-threshold",
        type=float,
        default=DEFAULT_AUTO_PAUSE_THRESHOLD,
        help="Per-action-dimension movement threshold before logging a new step.",
    )
    parser.add_argument(
        "--grip-hold-sec",
        type=float,
        default=DEFAULT_GRIP_HOLD_SEC,
        help="Seconds to continue stepping after a gripper close toggle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        task=args.task,
        manipulator=args.manipulator,
        end_effector=args.end_effector,
        dataset_root=args.dataset_root,
        controller_id=args.controller_id,
        controller_side=args.controller_side,
        controller_source=args.controller_source,
        controller_transport=args.controller_transport,
        controller_endpoint=args.controller_endpoint,
        controller_topic=args.controller_topic,
        controller_discovery_timeout_sec=args.controller_discovery_timeout_sec,
        controller_recv_timeout_ms=args.controller_recv_timeout_ms,
        controller_stale_timeout_sec=args.controller_stale_timeout_sec,
        haptics=args.haptics,
        auto_pause_threshold=args.auto_pause_threshold,
        grip_hold_sec=args.grip_hold_sec,
    )
