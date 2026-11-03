from __future__ import annotations

import argparse
import contextlib
import time

from .vive_controller_zmq import ViveControllerZMQServer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish SteamVR Vive controller pose/buttons over ZMQ.",
    )
    parser.add_argument(
        "--bind",
        default="tcp://*:5558",
        help="ZMQ PUB bind endpoint.",
    )
    parser.add_argument(
        "--topic",
        default="vive.controller",
        help="ZMQ topic used for controller snapshots.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "any"),
        default="left",
        help="Controller side to publish when not using --publish-all.",
    )
    parser.add_argument(
        "--controller-id",
        default="controller_1",
        help="Fallback OpenVR controller name or serial.",
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
        help="Publish every discovered OpenVR controller.",
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

    server = ViveControllerZMQServer(
        controller_id=args.controller_id,
        side=side,
        bind_endpoint=args.bind,
        topic=args.topic,
        rate_hz=args.rate_hz,
        publish_all=args.publish_all,
        poll_events=args.poll_events,
    )

    print(
        f"Publishing Vive controller data on {args.bind} [{args.topic}].",
        flush=True,
    )
    if args.publish_all:
        print("Publishing all discovered Vive controllers.", flush=True)
    else:
        print(
            f"Publishing side={args.side!r}, controller_id={args.controller_id!r}.",
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
                if published_count > 0:
                    device_labels = [
                        f"{name}({','.join(device_formats.get(name, ())) or 'empty'})"
                        for name in device_names
                    ]
                    print(
                        f"[VIVE-ZMQ] Publishing {published_count} controller(s): "
                        + ", ".join(device_labels),
                        flush=True,
                    )
                else:
                    print(
                        "[VIVE-ZMQ] No valid controller pose available yet.",
                        flush=True,
                    )
                last_status = (published_count, device_names, format_status)
                last_status_t = now
            if period_sec > 0.0:
                time.sleep(period_sec)
    except KeyboardInterrupt:
        print("\nStopping Vive controller ZMQ server...", flush=True)
    finally:
        with contextlib.suppress(Exception):
            server.close()


if __name__ == "__main__":
    main()
