from __future__ import annotations

from pathlib import Path

import genesis as gs
import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _init_genesis():
    try:
        gs.init()
    except gs.GenesisException as exc:
        if "already initialized" not in str(exc):
            raise


@pytest.fixture(autouse=True)
def _disable_offscreen_rasterizer(monkeypatch):
    """Keep controller physics tests independent from EGL/OpenGL availability."""
    from genesis.vis.rasterizer import Rasterizer

    def build_without_renderer(self):
        if self._context is not None:
            self.visualizer = self._context.visualizer

    monkeypatch.setattr(Rasterizer, "build", build_without_renderer)


@pytest.fixture(scope="function", params=[1, 4], ids=lambda n: f"n_envs={n}")
def scene_robot(request, monkeypatch):
    n_envs: int = int(request.param)
    panda_xml = (
        Path(gs.__file__).resolve().parent
        / "assets"
        / "xml"
        / "franka_emika_panda"
        / "panda.xml"
    )

    scene = gs.Scene(show_viewer=False)
    _ = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    # Build single or parallel scene
    if n_envs == 1:
        scene.build()
    else:
        scene.build(n_envs=n_envs, env_spacing=(0.7, 0.7))

    # Patch the functions actually used by Controller (imported into that module)
    import importlib

    bc_mod = importlib.import_module("dexsuite.controllers.base_controller")
    monkeypatch.setattr(bc_mod, "get_n_envs", lambda: n_envs, raising=False)
    monkeypatch.setattr(
        bc_mod,
        "get_device",
        lambda: torch.device(gs.device),
        raising=False,
    )

    ee = robot.get_link("hand")
    qpos = robot.get_dofs_position()
    dof = int(qpos.shape[-1])

    class FakeArmModel:
        model_path: str = str(panda_xml)
        end_link: str = "hand"
        root_link: str = "link0"

        control_dofs_index = None

        def __init__(self, dof_in: int) -> None:
            self.dof = int(dof_in)
            self.home_q = np.zeros(self.dof, dtype=np.float32)

            # Position limits (fallback to ±π)
            try:
                lo, hi = robot.get_dofs_limit()
                if lo.ndim == 2:
                    lo = lo[0]
                    hi = hi[0]
                self.q_lo = lo.detach().cpu().numpy().astype(np.float32)
                self.q_hi = hi.detach().cpu().numpy().astype(np.float32)
            except Exception:
                self.q_lo = -np.pi * np.ones(self.dof, np.float32)
                self.q_hi = +np.pi * np.ones(self.dof, np.float32)

            # Gains (arbitrary but consistent)
            self.kp = np.full(self.dof, 4000.0, np.float32)
            self.kv = np.full(self.dof, 40.0, np.float32)

            # Force ranges (fallback to 30 Nm)
            try:
                flo, fhi = robot.get_dofs_force_range()
                if flo.ndim == 2:
                    flo = flo[0]
                    fhi = fhi[0]
                fr = np.maximum(
                    np.abs(flo.detach().cpu().numpy()),
                    np.abs(fhi.detach().cpu().numpy()),
                ).astype(np.float32)
                self.force_range = fr
            except Exception:
                self.force_range = np.full(self.dof, 30.0, np.float32)

    model = FakeArmModel(dof)
    yield scene, robot, ee, model, n_envs
