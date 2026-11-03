"""Curses-based TUI for building DexSuite environments.

Key bindings (main screen)
--------------------------
- ↑/↓: Navigate fields
- Enter: Edit selected field
- q: Quit (keep current spec)
- Done: Finish and return spec
"""

from __future__ import annotations

import contextlib
from dataclasses import replace

if __package__:
    from scripts.interactive_builder.defaults import default_spec
    from scripts.interactive_builder.registry_scan import (
        available_controllers,
        available_grippers,
        available_manipulators,
        available_static_cameras,
        available_tasks,
        manipulator_infos,
        supported_grippers_for_manipulator,
    )
    from scripts.interactive_builder.spec import (
        POSE_BACKENDS,
        ArmSpec,
        BuilderSpec,
        arm_backend,
        controller_supports_backend,
        set_arm_backend,
    )
else:
    from defaults import default_spec
    from registry_scan import (
        available_controllers,
        available_grippers,
        available_manipulators,
        available_static_cameras,
        available_tasks,
        manipulator_infos,
        supported_grippers_for_manipulator,
    )
    from spec import (
        POSE_BACKENDS,
        ArmSpec,
        BuilderSpec,
        arm_backend,
        controller_supports_backend,
        set_arm_backend,
    )


def run_tui() -> BuilderSpec:
    """Launch the curses UI and return a BuilderSpec."""
    try:
        import curses
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "curses is not available on this platform; use --ui simple.",
        ) from e

    spec0 = default_spec()
    out: dict[str, BuilderSpec] = {"spec": spec0}

    def _main(stdscr) -> None:
        out["spec"] = _BuilderTUI(stdscr, spec0).run()

    curses.wrapper(_main)
    return out["spec"]


class _BuilderTUI:
    def __init__(self, stdscr, spec: BuilderSpec) -> None:
        import curses

        self.curses = curses
        self.stdscr = stdscr
        self.spec = spec

        self.tasks = available_tasks()
        self.manipulators = available_manipulators()
        self.grippers = available_grippers()
        self.controllers = available_controllers()
        self.static_cameras = available_static_cameras()

        self.manip_kind = {i.key: i.kind for i in manipulator_infos()}

        self._init_screen()

    # --------------------------- curses helpers ---------------------------
    def _init_screen(self) -> None:
        c = self.curses
        self.stdscr.nodelay(False)  # noqa: FBT003
        self.stdscr.keypad(True)  # noqa: FBT003
        with contextlib.suppress(c.error):
            c.curs_set(0)
        if c.has_colors():
            c.start_color()
            c.use_default_colors()
            c.init_pair(1, c.COLOR_CYAN, -1)  # title
            c.init_pair(2, c.COLOR_BLACK, c.COLOR_WHITE)  # selection
            c.init_pair(3, c.COLOR_YELLOW, -1)  # hint
            c.init_pair(4, c.COLOR_RED, -1)  # error/warn

    def _color(self, pair: int) -> int:
        try:
            return self.curses.color_pair(pair)
        except self.curses.error:
            return 0

    def _draw_center(self, y: int, s: str, attr: int = 0) -> None:
        _height, w = self.stdscr.getmaxyx()
        x = max(0, (w - len(s)) // 2)
        with contextlib.suppress(self.curses.error):
            self.stdscr.addstr(y, x, s[: max(0, w - 1)], attr)

    def _prompt(self, prompt: str, default: str = "") -> str:
        c = self.curses
        h, w = self.stdscr.getmaxyx()
        y = h - 2
        self.stdscr.move(y, 0)
        self.stdscr.clrtoeol()
        s = f"{prompt} [{default}]: "
        self.stdscr.addstr(y, 0, s[: max(0, w - 1)], self._color(3))
        self.stdscr.refresh()

        c.echo()
        try:
            raw = self.stdscr.getstr(y, min(len(s), w - 1)).decode(
                "utf-8",
                errors="ignore",
            )
        finally:
            c.noecho()
        raw = raw.strip()
        return default if raw == "" else raw

    def _select_one(self, title: str, options: list[str], current: str | None) -> str:
        c = self.curses
        idx = 0
        if current in options:
            idx = options.index(current)
        while True:
            self.stdscr.clear()
            self._draw_center(0, title, self._color(1))
            self._draw_center(
                1,
                "↑/↓ navigate | Enter select | q cancel",
                self._color(3),
            )

            h, w = self.stdscr.getmaxyx()
            top = 3
            view_h = max(1, h - top - 2)
            start = max(0, idx - view_h // 2)
            end = min(len(options), start + view_h)
            start = max(0, end - view_h)

            for row, opt_i in enumerate(range(start, end), start=top):
                opt = options[opt_i]
                attr = self._color(2) if opt_i == idx else 0
                self.stdscr.addstr(row, 2, opt[: max(0, w - 4)], attr)

            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (ord("q"), 27):
                return current if current is not None else options[0]
            if k in (c.KEY_UP, ord("k")):
                idx = (idx - 1) % len(options)
            elif k in (c.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(options)
            elif k in (c.KEY_ENTER, 10, 13):
                return options[idx]

    def _select_multi(
        self,
        title: str,
        options: list[str],
        selected: set[str],
    ) -> set[str]:
        c = self.curses
        idx = 0
        selected = set(selected)
        while True:
            self.stdscr.clear()
            self._draw_center(0, title, self._color(1))
            self._draw_center(
                1,
                "↑/↓ navigate | Space toggle | Enter done | q cancel",
                self._color(3),
            )

            h, w = self.stdscr.getmaxyx()
            top = 3
            view_h = max(1, h - top - 2)
            start = max(0, idx - view_h // 2)
            end = min(len(options), start + view_h)
            start = max(0, end - view_h)

            for row, opt_i in enumerate(range(start, end), start=top):
                opt = options[opt_i]
                mark = "[x]" if opt in selected else "[ ]"
                line = f"{mark} {opt}"
                attr = self._color(2) if opt_i == idx else 0
                self.stdscr.addstr(row, 2, line[: max(0, w - 4)], attr)

            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (ord("q"), 27):
                return selected
            if k in (c.KEY_UP, ord("k")):
                idx = (idx - 1) % len(options)
            elif k in (c.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(options)
            elif k == ord(" "):
                opt = options[idx]
                if opt in selected:
                    selected.remove(opt)
                else:
                    selected.add(opt)
            elif k in (c.KEY_ENTER, 10, 13):
                return selected

    # ------------------------------ edits ------------------------------
    def _ensure_arms(self) -> None:
        if self.spec.type_of_robot == "single":
            if self.spec.single is None:
                self.spec.single = ArmSpec(
                    manipulator="franka",
                    gripper="robotiq",
                    arm_control="osc_pose",
                    gripper_control="joint_position",
                )
        else:
            if self.spec.left is None or self.spec.right is None:
                base = self.spec.single or ArmSpec(
                    manipulator="franka",
                    gripper="robotiq",
                    arm_control="osc_pose",
                    gripper_control="joint_position",
                )
                self.spec.left = replace(base)
                self.spec.right = replace(base)
            self.spec.single = None

    def _edit_task(self) -> None:
        if not self.tasks:
            return
        self.spec.task = self._select_one("Select Task", self.tasks, self.spec.task)

    def _edit_robot_type(self) -> None:
        chosen = self._select_one(
            "Select Robot Type",
            ["single", "bimanual"],
            self.spec.type_of_robot,
        )
        self.spec.type_of_robot = chosen  # type: ignore[assignment]
        if self.spec.type_of_robot == "single":
            self.spec.layout_preset = None
        elif not self.spec.layout_preset:
            self.spec.layout_preset = "side_by_side"
        self._ensure_arms()

    def _edit_manipulator(self, arm: str) -> None:
        self._ensure_arms()
        a = self._get_arm(arm)
        if not self.manipulators:
            return
        prev = a.manipulator
        a.manipulator = self._select_one(
            f"Select {arm} manipulator",
            self.manipulators,
            a.manipulator,
        )
        if self.manip_kind.get(a.manipulator) == "integrated":
            a.gripper = None
            return
        # If we changed manipulators, ensure the gripper still makes sense.
        if a.manipulator != prev:
            supported = supported_grippers_for_manipulator(a.manipulator)
            if supported and a.gripper not in supported:
                a.gripper = supported[0]

    def _edit_gripper(self, arm: str) -> None:
        self._ensure_arms()
        a = self._get_arm(arm)
        if self.manip_kind.get(a.manipulator) == "integrated":
            a.gripper = None
            return
        supported = supported_grippers_for_manipulator(a.manipulator)
        options = supported if supported else self.grippers
        if not options:
            return
        a.gripper = self._select_one(f"Select {arm} gripper", options, a.gripper)

    def _edit_arm_ctrl(self, arm: str) -> None:
        self._ensure_arms()
        a = self._get_arm(arm)
        if not self.controllers:
            return
        prev = a.arm_control
        a.arm_control = self._select_one(
            f"Select {arm} arm controller",
            self.controllers,
            a.arm_control,
        )
        if a.arm_control != prev:
            a.arm_control_config.clear()
            set_arm_backend(a, POSE_BACKENDS[0])

    def _edit_arm_backend(self, arm: str) -> None:
        self._ensure_arms()
        a = self._get_arm(arm)
        if not controller_supports_backend(a.arm_control):
            set_arm_backend(a, POSE_BACKENDS[0])
            return
        chosen = self._select_one(
            f"Select {arm} arm controller backend",
            list(POSE_BACKENDS),
            arm_backend(a),
        )
        set_arm_backend(a, chosen)

    def _edit_grip_ctrl(self, arm: str) -> None:
        self._ensure_arms()
        a = self._get_arm(arm)
        options = [c for c in self.controllers if c.startswith("joint_")]
        if not options:
            return
        cur = a.gripper_control or options[0]
        prev = a.gripper_control
        a.gripper_control = self._select_one(
            f"Select {arm} gripper controller",
            options,
            cur,
        )
        if a.gripper_control != prev:
            a.gripper_control_config.clear()

    def _edit_layout_preset(self) -> None:
        if self.spec.type_of_robot != "bimanual":
            return
        # Keep it simple: offer the built-in layouts from layouts/defaults.yaml.
        presets = ["side_by_side", "face_to_face", "right_angle"]
        cur = self.spec.layout_preset or "side_by_side"
        chosen = self._select_one("Select bimanual layout preset", presets, cur)
        self.spec.layout_preset = chosen

    def _edit_int(self, field: str, *, lo: int = 1, hi: int = 10_000) -> None:
        cur = getattr(self.spec, field)
        raw = self._prompt(f"{field}", str(cur))
        try:
            v = int(raw)
        except ValueError:
            return
        v = max(lo, min(hi, v))
        setattr(self.spec, field, v)

    def _edit_bool(self, field: str) -> None:
        cur = bool(getattr(self.spec, field))
        raw = self._prompt(f"{field} (0/1)", "1" if cur else "0")
        try:
            setattr(self.spec, field, bool(int(raw)))
        except ValueError:
            return

    def _edit_render_mode(self) -> None:
        opts = ["human", "rgb_array", "none"]
        cur = self.spec.render_mode or "none"
        chosen = self._select_one("Select render_mode", opts, cur)
        self.spec.render_mode = None if chosen == "none" else chosen  # type: ignore[assignment]

    def _edit_input_device(self) -> None:
        if self.spec.type_of_robot == "bimanual":
            # Keep it simple for now (runner is single-arm teleop only).
            self.spec.input_device = "none"
            return
        opts = ["keyboard", "spacemouse", "vive_controller", "vive_tracker", "none"]
        cur = self.spec.input_device or "keyboard"
        self.spec.input_device = self._select_one("Select input device", opts, cur)

    def _edit_modalities(self) -> None:
        opts = ["rgb", "depth", "segmentation", "normal"]
        sel = set(self.spec.modalities)
        sel = self._select_multi("Select modalities (rgb required)", opts, sel)
        sel.add("rgb")
        self.spec.modalities = tuple(sorted(sel))

    def _edit_cameras(self) -> None:
        # Represent cameras in 3 states:
        # - None: defaults
        # - []: none
        # - list: explicit
        opts = ["<defaults>", "<none>", "wrist", *self.static_cameras]
        cur = self.spec.cameras
        if cur is None:
            selected: set[str] = {"<defaults>"}
        elif len(cur) == 0:
            selected = {"<none>"}
        else:
            selected = set(cur)

        selected = self._select_multi("Select cameras", opts, selected)
        if "<defaults>" in selected:
            self.spec.cameras = None
            return
        if "<none>" in selected:
            self.spec.cameras = []
            return
        selected.discard("<defaults>")
        selected.discard("<none>")
        self.spec.cameras = sorted(selected)

    def _get_arm(self, which: str) -> ArmSpec:
        if self.spec.type_of_robot == "single":
            if self.spec.single is None:
                raise ValueError("Single robot is missing its arm spec.")
            return self.spec.single
        if which.lower() == "left":
            if self.spec.left is None:
                raise ValueError("Bimanual robot is missing its left arm spec.")
            return self.spec.left
        if which.lower() == "right":
            if self.spec.right is None:
                raise ValueError("Bimanual robot is missing its right arm spec.")
            return self.spec.right
        raise ValueError(f"Unknown arm: {which!r}")

    # ------------------------------ main loop ------------------------------
    def run(self) -> BuilderSpec:
        self._ensure_arms()
        idx = 0
        while True:
            items = self._menu_items()
            idx = min(idx, len(items) - 1)

            self.stdscr.clear()
            self._draw_center(0, "DexSuite Interactive Builder", self._color(1))
            self._draw_center(1, "Enter=edit | q=quit | Done=finish", self._color(3))

            h, w = self.stdscr.getmaxyx()
            top = 3
            for i, (label, value, _fn) in enumerate(items):
                y = top + i
                if y >= h - 2:
                    break
                line = f"{label:<20} {value}"
                attr = self._color(2) if i == idx else 0
                self.stdscr.addstr(y, 2, line[: max(0, w - 4)], attr)

            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (ord("q"), 27):
                return self.spec
            if k in (self.curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(items)
            elif k in (self.curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(items)
            elif k in (self.curses.KEY_ENTER, 10, 13):
                _, _, fn = items[idx]
                if fn is None:
                    return self.spec
                fn()

    def _menu_items(self):
        cam_val = (
            "default"
            if self.spec.cameras is None
            else (
                "none" if len(self.spec.cameras) == 0 else ",".join(self.spec.cameras)
            )
        )
        items: list[tuple[str, str, object]] = [
            ("Task", self.spec.task, self._edit_task),
            ("Robot type", self.spec.type_of_robot, self._edit_robot_type),
        ]

        if self.spec.type_of_robot == "single":
            a = self._get_arm("single")
            items += [
                (
                    "Manipulator",
                    a.manipulator,
                    lambda: self._edit_manipulator("Single"),
                ),
                (
                    "Gripper",
                    str(a.gripper) if a.gripper is not None else "builtin",
                    lambda: self._edit_gripper("Single"),
                ),
                ("Arm ctrl", a.arm_control, lambda: self._edit_arm_ctrl("Single")),
                (
                    "Arm backend",
                    arm_backend(a) or "n/a",
                    lambda: self._edit_arm_backend("Single"),
                ),
                (
                    "Grip ctrl",
                    str(a.gripper_control) if a.gripper_control else "none",
                    lambda: self._edit_grip_ctrl("Single"),
                ),
            ]
        else:
            la = self._get_arm("left")
            ra = self._get_arm("right")
            items += [
                (
                    "Layout preset",
                    str(self.spec.layout_preset)
                    if self.spec.layout_preset
                    else "side_by_side",
                    self._edit_layout_preset,
                ),
                ("L manip", la.manipulator, lambda: self._edit_manipulator("Left")),
                (
                    "L gripper",
                    str(la.gripper) if la.gripper is not None else "builtin",
                    lambda: self._edit_gripper("Left"),
                ),
                ("L arm ctrl", la.arm_control, lambda: self._edit_arm_ctrl("Left")),
                (
                    "L backend",
                    arm_backend(la) or "n/a",
                    lambda: self._edit_arm_backend("Left"),
                ),
                (
                    "L grip ctrl",
                    str(la.gripper_control) if la.gripper_control else "none",
                    lambda: self._edit_grip_ctrl("Left"),
                ),
                ("R manip", ra.manipulator, lambda: self._edit_manipulator("Right")),
                (
                    "R gripper",
                    str(ra.gripper) if ra.gripper is not None else "builtin",
                    lambda: self._edit_gripper("Right"),
                ),
                ("R arm ctrl", ra.arm_control, lambda: self._edit_arm_ctrl("Right")),
                (
                    "R backend",
                    arm_backend(ra) or "n/a",
                    lambda: self._edit_arm_backend("Right"),
                ),
                (
                    "R grip ctrl",
                    str(ra.gripper_control) if ra.gripper_control else "none",
                    lambda: self._edit_grip_ctrl("Right"),
                ),
            ]

        items += [
            (
                "control_hz",
                str(self.spec.control_hz),
                lambda: self._edit_int("control_hz", lo=1, hi=1000),
            ),
            (
                "n_envs",
                str(self.spec.n_envs),
                lambda: self._edit_int("n_envs", lo=1, hi=16384),
            ),
            (
                "perf_mode",
                str(int(self.spec.performance_mode)),
                lambda: self._edit_bool("performance_mode"),
            ),
            ("input", str(self.spec.input_device), self._edit_input_device),
            ("cameras", cam_val, self._edit_cameras),
            ("modalities", ",".join(self.spec.modalities), self._edit_modalities),
            (
                "render_mode",
                str(self.spec.render_mode) if self.spec.render_mode else "none",
                self._edit_render_mode,
            ),
            ("Done", "", None),
        ]
        return items
