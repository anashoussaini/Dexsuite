from __future__ import annotations

import argparse
import contextlib
import time

from .vive_tracker_zmq import ViveTrackerZMQServer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish SteamVR Vive tracker pose over ZMQ.",
    )
    parser.add_argument(
        "--bind",
        default="tcp://*:5559",
        help="ZMQ PUB bind endpoint.",
    )
    parser.add_argument(
        "--topic",
        default="vive.tracker",
        help="ZMQ topic used for tracker snapshots.",
    )
    parser.add_argument(
        "--side",
        default="center",
        help="Tracker role label to publish when not using --publish-all.",
    )
    parser.add_argument(
        "--tracker-id",
        default="tracker_1",
        help="Fallback OpenVR tracker name or serial.",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=120.0,
        help=(
            "Publish rate for outbound snapshots. "
            "Use <=0 to publish as fast as possible."
        ),
    )
    parser.add_argument(
        "--publish-all",
        action="store_true",
        help="Publish every discovered OpenVR tracker.",
    )
    parser.add_argument(
        "--no-poll-events",
        action="store_false",
        dest="poll_events",
        help="Disable OpenVR event polling while publishing.",
    )
    parser.add_argument(
        "--status-every-sec",
        type=float,
        default=1.0,
        help="Print publisher status at this interval. Use <=0 to print only changes.",
    )
    parser.set_defaults(poll_events=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    side = "" if args.side == "any" else args.side
    period_sec = 0.0 if args.rate_hz <= 0.0 else (1.0 / float(args.rate_hz))

    server = ViveTrackerZMQServer(
        tracker_id=args.tracker_id,
        side=side,
        bind_endpoint=args.bind,
        topic=args.topic,
        rate_hz=args.rate_hz,
        publish_all=args.publish_all,
        poll_events=args.poll_events,
    )

    print(
        f"Publishing Vive tracker data on {args.bind} [{args.topic}].",
        flush=True,
    )
    if args.publish_all:
        print("Publishing all discovered Vive trackers.", flush=True)
    else:
        print(
            f"Publishing side={args.side!r}, tracker_id={args.tracker_id!r}.",
            flush=True,
        )
    print("Press Ctrl+C to stop.", flush=True)

    last_status: (
        tuple[
            int,
            tuple[str, ...],
            tuple[tuple[str, tuple[str, ...]], ...],
        ]
        | None
    ) = None
    last_status_t = 0.0
    try:
        while True:
            published_count = server.publish_once()
            device_names = server.last_published_device_names
            device_formats = server.last_published_device_formats
            format_status = tuple(sorted(device_formats.items()))
            now = time.time()
            should_print = (
                published_count,
                device_names,
                format_status,
            ) != last_status
            if (
                args.status_every_sec > 0.0
                and (now - last_status_t) >= args.status_every_sec
            ):
                should_print = True
            if should_print:
                all_devices = tuple(sorted(server.vr.devices))
                tracker_items = server._tracker_items()
                publish_devices = server._publish_devices()
                print(
                    "[VIVE-TRACKER-ZMQ][debug] "
                    f"all_openvr_devices={all_devices} "
                    f"tracker_candidates={[name for name, _ in tracker_items]} "
                    f"selected_publish_names={tuple(sorted(publish_devices))}",
                    flush=True,
                )
                for name, dev in tracker_items:
                    published_as = next(
                        (
                            pub_name
                            for pub_name, pub_dev in publish_devices.items()
                            if pub_dev is dev
                        ),
                        None,
                    )
                    try:
                        serial = getattr(dev, "get_serial", lambda: name)()
                    except Exception as exc:
                        serial = f"<serial error: {exc}>"
                    try:
                        pose = dev.get_pose_quaternion()
                    except Exception as exc:
                        pose = f"<pose error: {exc}>"
                    try:
                        buttons = dev.get_controller_inputs() or {}
                    except Exception as exc:
                        buttons = f"<buttons error: {exc}>"
                    print(
                        "[VIVE-TRACKER-ZMQ][debug] "
                        f"tracker={name!r} published_as={published_as!r} "
                        f"class={getattr(dev, 'device_class', None)!r} "
                        f"index={getattr(dev, 'index', None)!r} "
                        f"serial={serial!r} pose={pose!r} buttons={buttons!r}",
                        flush=True,
                    )
                if published_count > 0:
                    device_labels = [
                        f"{name}({','.join(device_formats.get(name, ())) or 'empty'})"
                        for name in device_names
                    ]
                    print(
                        f"[VIVE-TRACKER-ZMQ] Publishing {published_count} "
                        "tracker(s): " + ", ".join(device_labels),
                        flush=True,
                    )
                else:
                    print(
                        "[VIVE-TRACKER-ZMQ] No valid tracker pose available yet.",
                        flush=True,
                    )
                last_status = (published_count, device_names, format_status)
                last_status_t = now
            if period_sec > 0.0:
                time.sleep(period_sec)
    except KeyboardInterrupt:
        print("\nStopping Vive tracker ZMQ server...", flush=True)
    finally:
        with contextlib.suppress(Exception):
            server.close()


if __name__ == "__main__":
    main()
