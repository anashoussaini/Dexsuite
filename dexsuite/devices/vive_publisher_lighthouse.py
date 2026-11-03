from __future__ import annotations

import argparse
import contextlib
import time

from .survive_utils import survive_context
from .tracker_zmq import tracker_zmq_publisher


def _is_publishable_device(name: str) -> bool:
    label = name.lower()
    if any(token in label for token in ("hmd", "lighthouse", "base")):
        return False
    if label.startswith(("lh0", "lh1", "lh2", "lhb")):
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Lighthouse device state via pysurvive and publish it over ZMQ.",
    )
    parser.add_argument(
        "--bind",
        default="tcp://*:5558",
        help="ZMQ PUB bind endpoint.",
    )
    parser.add_argument(
        "--topic",
        default="lighthouse.poses",
        help="ZMQ topic used for device snapshots.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=120.0,
        help="Publish rate for outbound snapshots.",
    )
    parser.add_argument(
        "--discovery-timeout-sec",
        type=float,
        default=3.0,
        help="How long to wait for initial libsurvive object discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    period_sec = 0.0 if args.rate_hz <= 0.0 else (1.0 / float(args.rate_hz))
    last_status: tuple[int, tuple[str, ...]] | None = None
    debug_dumped = False

    vr = survive_context(discovery_timeout_sec=args.discovery_timeout_sec)
    publisher = tracker_zmq_publisher(
        bind_endpoint=args.bind,
        topic=args.topic,
    )

    print(
        f"Using libsurvive backend: {getattr(vr, '_backend', 'unknown')}.",
        flush=True,
    )
    print(
        f"Publishing Lighthouse device data on {args.bind} [{args.topic}].", flush=True
    )
    print("Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            with vr._device_lock:
                devices = {
                    name: dev
                    for name, dev in vr.devices.items()
                    if _is_publishable_device(str(name))
                }
            published_count = publisher.publish_snapshot(devices)
            device_names = tuple(sorted(str(name) for name in devices))
            status = (published_count, device_names)
            if status != last_status:
                if published_count > 0:
                    print(
                        f"[PUBLISHER] Publishing {published_count} device(s): "
                        + ", ".join(device_names),
                        flush=True,
                    )
                else:
                    if device_names:
                        print(
                            "[PUBLISHER] Discovered devices but no valid poses yet: "
                            + ", ".join(device_names),
                            flush=True,
                        )
                        if not debug_dumped:
                            for name, dev in devices.items():
                                if str(name).startswith("LH"):
                                    continue
                                raw_obj = getattr(dev, "raw_object", None)
                                pose_obj = None
                                if raw_obj is not None:
                                    pose_attr = getattr(raw_obj, "Pose", None)
                                    if callable(pose_attr):
                                        with contextlib.suppress(Exception):
                                            pose_obj = pose_attr()
                                    elif pose_attr is not None:
                                        pose_obj = pose_attr
                                print(
                                    "[PUBLISHER][DEBUG] "
                                    f"{name}: raw_type={type(raw_obj).__name__}, "
                                    f"pose_type={type(pose_obj).__name__}, "
                                    f"pose_repr={repr(pose_obj)}",
                                    flush=True,
                                )
                            debug_dumped = True
                    else:
                        print(
                            "[PUBLISHER] No valid Lighthouse device data available yet.",
                            flush=True,
                        )
                last_status = status
            if period_sec > 0.0:
                time.sleep(period_sec)
    except KeyboardInterrupt:
        print("\nStopping publisher...", flush=True)
    finally:
        with contextlib.suppress(Exception):
            publisher.close()
        with contextlib.suppress(Exception):
            vr.close()


if __name__ == "__main__":
    main()
