"""Collect LeRobot episodes with Vive Tracker ZMQ and Manus hand input.

This script initializes a single-arm task, captures arm pose input from a
remote Vive Tracker ZMQ stream, reads Manus hand qpos over WebSocket, and saves
successful episodes with LeRobotLogger.

Key Controls:
- Vive Tracker: Controls the end-effector pose.
- Manus/GeoRT WebSocket stream: Controls the Allegro hand joints.
- Keyboard reset: Pressing the reset key drops the current trajectory and
    resets the environment without saving.

Episode Logic:
- Only episodes marked with 'success' are saved.
- Failed, terminated, or truncated episodes are dropped.
- A KeyboardInterrupt (Ctrl+C) will stop the script and finalize the
    dataset, discarding any in-progress (partial) episode.
"""

import argparse
import contextlib
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import dexsuite as ds
from dexsuite.devices import ZMQViveTracker
from dexsuite.devices.external_reset_helper import ExternalResetHelper
from dexsuite.devices.manus_geort import ManusGeort
from dexsuite.utils.lerobot_logger import LeRobotLogger

CTRL_HZ = 20
DEFAULT_TRACKER_ENDPOINT = "tcp://127.0.0.1:5559"
DEFAULT_TRACKER_TOPIC = "vive.tracker"
DEFAULT_TRACKER_ID = "tracker_1"
DEFAULT_TRACKER_SIDE = "center"
DEFAULT_TRACKER_DISCOVERY_TIMEOUT_SEC = 3.0
DEFAULT_MANUS_URL = "ws://127.0.0.1:8767"
DEFAULT_MANIPULATOR = "ur5"
DEFAULT_END_EFFECTOR = "allegro"
DEFAULT_DATASET_ROOT = "collected_dataset"
DEFAULT_DEBUG_EVERY_SEC = -1.0


def get_device() -> torch.device:
    """Determine the torch device used for action tensors.

    Returns:
        The CUDA device when available, otherwise the CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_bool(val) -> bool:
    """Convert a tensor-like or scalar value to a Python boolean.

    Args:
        val: Value to convert. Supports scalar tensors, arrays, booleans, and
            integers.

    Returns:
        The Python boolean representation of ``val``.
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


def _flat_obs_tensor(obs: dict, *keys: str) -> torch.Tensor | None:
    """Read a nested observation value as a flattened CPU float tensor.

    Args:
        obs: Observation dictionary returned by the environment.
        *keys: Nested keys to traverse before converting the value.

    Returns:
        A flattened ``torch.float32`` CPU tensor when the key path exists and
        can be converted, otherwise ``None``.
    """
    value = obs
    try:
        for key in keys:
            value = value[key]
    except (KeyError, TypeError):
        return None

    if torch.is_tensor(value):
        return value.detach().float().reshape(-1).cpu()
    try:
        return torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    except (TypeError, ValueError):
        return None


def _fmt_vec3(value: torch.Tensor | np.ndarray | None) -> str:
    """Format a vector-like value for compact debug logging.

    Args:
        value: Tensor or array with at least three elements, or ``None``.

    Returns:
        A ``(+x,+y,+z)`` string when available, otherwise ``"(none)"``.
    """
    if value is None or len(value) < 3:
        return "(none)"
    return f"({float(value[0]):+.3f},{float(value[1]):+.3f},{float(value[2]):+.3f})"


def _fmt_delta(value: float | None) -> str:
    """Format an optional scalar delta for debug logging.

    Args:
        value: Scalar delta value, or ``None`` when no previous sample exists.

    Returns:
        A four-decimal string for real values, otherwise ``"none"``.
    """
    return "none" if value is None else f"{value:.4f}"


def main(
    *,
    task: str = "pick_place_fruit",
    manipulator: str = DEFAULT_MANIPULATOR,
    end_effector: str = DEFAULT_END_EFFECTOR,
    dataset_root: str = DEFAULT_DATASET_ROOT,
    tracker_endpoint: str = DEFAULT_TRACKER_ENDPOINT,
    tracker_topic: str = DEFAULT_TRACKER_TOPIC,
    tracker_id: str = DEFAULT_TRACKER_ID,
    tracker_side: str = DEFAULT_TRACKER_SIDE,
    tracker_discovery_timeout_sec: float = DEFAULT_TRACKER_DISCOVERY_TIMEOUT_SEC,
    manus_url: str = DEFAULT_MANUS_URL,
    debug_every_sec: float = DEFAULT_DEBUG_EVERY_SEC,
):
    """Run Manus hand data collection with a ZMQ-backed Vive tracker.

    Args:
        task: DexSuite task name to instantiate.
        manipulator: Manipulator model used by the single-arm robot.
        end_effector: End-effector or gripper model used by the robot.
        dataset_root: Root directory for saved LeRobot datasets.
        tracker_endpoint: ZMQ endpoint serving Vive tracker pose samples.
        tracker_topic: ZMQ topic to subscribe to for tracker data.
        tracker_id: Remote tracker name or serial to select.
        tracker_side: Tracker role label used by the tracker client.
        tracker_discovery_timeout_sec: Seconds to wait for initial tracker
            samples during client setup.
        manus_url: Manus/GeoRT WebSocket URL serving hand qpos samples.
        debug_every_sec: Interval for diagnostic logging. Values less than or
            equal to zero disable periodic debug output.
    """
    env = ds.make(
        task,
        sim=ds.options.SimOptions(control_hz=20),
        robot=ds.options.RobotOptions(
            single=ds.options.ArmOptions(
                manipulator=manipulator,
                gripper=end_effector,
                manipulator_controller=ds.options.ControllerOptions(
                    name="osc_pose_abs_quat",
                ),
                gripper_controller=ds.options.ControllerOptions(
                    name="joint_position",
                    config={"normalized": False},
                ),
            ),
        ),
        cameras=["front", "wrist"],
        render_mode="human",
    )
    gripper_dim = int(env.robot.hand_ctrl.action_space().shape[0])
    dataset_dir = _collection_dataset_dir(
        dataset_root,
        task,
        manipulator,
        end_effector,
    )
    print(f"Saving collected dataset to {dataset_dir}", flush=True)

    # Tracker input
    print(
        "Connecting ZMQ Vive Tracker to "
        f"{tracker_endpoint} [{tracker_topic}] "
        f"tracker_id={tracker_id!r} side={tracker_side!r}.",
        flush=True,
    )
    vt = ZMQViveTracker(
        tracker_id=tracker_id,
        side=tracker_side,
        endpoint=tracker_endpoint,
        topic=tracker_topic,
        discovery_timeout_sec=tracker_discovery_timeout_sec,
    )
    print(f"Vive Tracker ZMQ status: {vt.bridge_status_text()}", flush=True)
    manus = ManusGeort(url=manus_url, expected_dim=gripper_dim)
    logger = LeRobotLogger(task=task, env=env, dataset_dir=dataset_dir)
    print(env.robot._layout, flush=True)

    trigger_keys = {"r": vt.recalibrate, "d": None, "k": None}   # k: save the attempt now (play data)
    ext_reset = ExternalResetHelper(trigger_keys)

    def get_gripper_action():
        """Return the latest Manus hand command in DexSuite action order.

        Returns:
            The current hand action vector sized for the configured gripper.
        """
        return manus.get_action()

    obs, info = env.reset()
    dev = get_device()
    count = 0
    last_debug_t = 0.0
    debug_last_raw_tracker_pos = None
    debug_last_grip_cmd = None
    debug_last_tcp_pos = None
    debug_last_grip_qpos = None
    try:
        try:
            while True:
                if ext_reset.key_pressed() == "d":
                    wait_for_reset(ext_reset, logger)
                    obs, info = env.reset()
                    continue

                tracker_action = vt.read()
                ax, ay, az, qw, qx, qy, qz = tracker_action["pose"]
                arm = torch.tensor(
                    [ax, ay, az, qw, qx, qy, qz],
                    dtype=torch.float32,
                    device=dev,
                )
                grip = torch.as_tensor(
                    get_gripper_action(),
                    dtype=torch.float32,
                    device=dev,
                )
                action_tensor = torch.cat([arm, grip], dim=0)

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
                now = time.time()
                if debug_every_sec > 0.0 and (now - last_debug_t) >= debug_every_sec:
                    needs_reset = bool(info.get("needs_reset", False))

                    raw_tracker_pose = None
                    tracker_status = {}
                    pose_status = getattr(
                        getattr(vt, "dev", None),
                        "pose_status",
                        None,
                    )
                    if callable(pose_status):
                        tracker_status = pose_status()
                    get_raw_pose = getattr(
                        getattr(vt, "dev", None),
                        "get_pose_quaternion",
                        None,
                    )
                    if callable(get_raw_pose):
                        raw_pose = get_raw_pose()
                        if raw_pose is not None:
                            raw_tracker_pose = np.asarray(
                                raw_pose[:3],
                                dtype=np.float32,
                            )
                    raw_tracker_delta = (
                        None
                        if raw_tracker_pose is None
                        or debug_last_raw_tracker_pos is None
                        else float(
                            np.linalg.norm(
                                raw_tracker_pose - debug_last_raw_tracker_pos,
                            ),
                        )
                    )

                    grip_cmd = grip.detach().float().reshape(-1).cpu()
                    grip_delta = (
                        None
                        if debug_last_grip_cmd is None
                        else float(torch.linalg.norm(grip_cmd - debug_last_grip_cmd))
                    )

                    obs_tcp_pos = _flat_obs_tensor(
                        next_obs,
                        "state",
                        "gripper",
                        "tcp_pos",
                    )
                    if obs_tcp_pos is not None:
                        obs_tcp_pos = obs_tcp_pos[:3]
                    obs_grip_qpos = _flat_obs_tensor(
                        next_obs,
                        "state",
                        "gripper",
                        "qpos",
                    )
                    obs_tcp_delta = (
                        None
                        if obs_tcp_pos is None or debug_last_tcp_pos is None
                        else float(torch.linalg.norm(obs_tcp_pos - debug_last_tcp_pos))
                    )
                    obs_grip_delta = (
                        None
                        if obs_grip_qpos is None or debug_last_grip_qpos is None
                        else float(
                            torch.linalg.norm(obs_grip_qpos - debug_last_grip_qpos),
                        )
                    )
                    obs_grip_norm = (
                        None
                        if obs_grip_qpos is None
                        else float(torch.linalg.norm(obs_grip_qpos))
                    )
                    tracker_age = tracker_status.get("age_sec")
                    tracker_age_text = (
                        "none" if tracker_age is None else f"{float(tracker_age):.3f}s"
                    )
                    print(
                        "[INPUT] "
                        f"tracker_pos=({ax:+.3f},{ay:+.3f},{az:+.3f}) "
                        f"tracker_q=({qw:+.3f},{qx:+.3f},{qy:+.3f},{qz:+.3f}) "
                        f"raw_tracker_pos={_fmt_vec3(raw_tracker_pose)} "
                        f"raw_tracker_delta={_fmt_delta(raw_tracker_delta)} "
                        f"tracker_fresh={tracker_status.get('fresh', 'n/a')} "
                        f"tracker_age={tracker_age_text} "
                        f"grip_norm={float(torch.linalg.norm(grip)):.3f} "
                        f"grip_delta={_fmt_delta(grip_delta)} "
                        f"obs_tcp={_fmt_vec3(obs_tcp_pos)} "
                        f"obs_tcp_delta={_fmt_delta(obs_tcp_delta)} "
                        f"obs_grip_norm={_fmt_delta(obs_grip_norm)} "
                        f"obs_grip_delta={_fmt_delta(obs_grip_delta)} "
                        f"action_norm={float(torch.linalg.norm(action_tensor)):.3f} "
                        f"success={success} done={bool(is_done)} "
                        f"needs_reset={needs_reset}",
                        flush=True,
                    )
                    if raw_tracker_pose is not None:
                        debug_last_raw_tracker_pos = raw_tracker_pose.copy()
                    debug_last_grip_cmd = grip_cmd.clone()
                    if obs_tcp_pos is not None:
                        debug_last_tcp_pos = obs_tcp_pos.clone()
                    if obs_grip_qpos is not None:
                        debug_last_grip_qpos = obs_grip_qpos.clone()
                    last_debug_t = now
                if success:
                    print(logger.ds.episode_buffer["size"])
                    count += 1
                    print(f"SUCCESS - saving episode {count}")

                    logger.save_episode()
                    print(f"SUCCESS — saved episode {count}")
                    wait_for_reset(ext_reset, logger)
                    obs = next_obs

                elif is_done:
                    print("END (failure/terminated/truncated) - dropping attempt")
                    # logger.clear_buffer()
                    wait_for_reset(ext_reset, logger)
                    obs = next_obs

                else:
                    obs = next_obs
        except KeyboardInterrupt:
            print("\nInterrupted by user. Not saving partial attempts (by design).")

        logger.close()
        print("Dataset finalized.")

    finally:
        with contextlib.suppress(Exception):
            vt.close()
        with contextlib.suppress(Exception):
            manus.close()
        with contextlib.suppress(Exception):
            ext_reset.close()
        env.close()
        print("Environment, Vive tracker, and Manus client closed.")


def wait_for_reset(reset_helper, logger):
    """Block until the operator recalibrates from the reset pose.

    Args:
        reset_helper: External reset helper that reports keyboard-triggered
            reset and recalibration events.
        logger: LeRobot logger whose in-progress episode buffer should be
            cleared after recalibration.
    """
    print("Episode: Please go back to origin position and recalibrate")
    while True:
        out = reset_helper.key_pressed()
        if out == "r":
            logger.clear_buffer()
            return


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Manus data collection.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect single-arm episodes with Vive Tracker ZMQ and Manus.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick_place_fruit",
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
        "--tracker-endpoint",
        type=str,
        default=DEFAULT_TRACKER_ENDPOINT,
        help="Vive tracker ZMQ endpoint, on the machine running the Vive publisher (e.g. tcp://<host>:5559).",
    )
    parser.add_argument(
        "--tracker-topic",
        type=str,
        default=DEFAULT_TRACKER_TOPIC,
        help="Vive tracker ZMQ topic to subscribe to.",
    )
    parser.add_argument(
        "--tracker-id",
        type=str,
        default=DEFAULT_TRACKER_ID,
        help="Remote Vive tracker name or serial.",
    )
    parser.add_argument(
        "--tracker-side",
        type=str,
        default=DEFAULT_TRACKER_SIDE,
        help="Tracker role label used for metadata and selection.",
    )
    parser.add_argument(
        "--tracker-discovery-timeout-sec",
        type=float,
        default=DEFAULT_TRACKER_DISCOVERY_TIMEOUT_SEC,
        help="Seconds to wait for initial Vive tracker ZMQ samples.",
    )
    parser.add_argument(
        "--manus-url",
        type=str,
        default=DEFAULT_MANUS_URL,
        help="Manus/GeoRT qpos WebSocket URL.",
    )
    parser.add_argument(
        "--debug-every-sec",
        type=float,
        default=DEFAULT_DEBUG_EVERY_SEC,
        help="Print input/action diagnostics at this interval. Use <=0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        task=args.task,
        manipulator=args.manipulator,
        end_effector=args.end_effector,
        dataset_root=args.dataset_root,
        tracker_endpoint=args.tracker_endpoint,
        tracker_topic=args.tracker_topic,
        tracker_id=args.tracker_id,
        tracker_side=args.tracker_side,
        tracker_discovery_timeout_sec=args.tracker_discovery_timeout_sec,
        manus_url=args.manus_url,
        debug_every_sec=args.debug_every_sec,
    )
