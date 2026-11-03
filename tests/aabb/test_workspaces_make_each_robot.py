from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

import dexsuite as ds
from dexsuite.core.components import IntegratedManipulatorModel
from dexsuite.core.registry import MANIP_REG
from dexsuite.core.robots.integrated_robot import IntegratedRobot
from dexsuite.core.robots.modular_robot import ModularRobot
from dexsuite.options import ArmOptions, RobotOptions
from dexsuite.utils.options_utils import load_workspaces

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _zeros_like(space: spaces.Space):
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=np.float32)
    if isinstance(space, spaces.Dict):
        return {k: _zeros_like(s) for k, s in space.spaces.items()}
    raise TypeError(f"Unsupported action space type: {type(space)}")


def _to_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


# Load the per-manipulator workspaces from YAML
_WORKSPACES: dict[str, dict[str, list[float]]] = load_workspaces()
_MANIPS = sorted(_WORKSPACES.keys())


@pytest.mark.parametrize("manip_name", _MANIPS)
@pytest.mark.slow
def test_env_world_aabb_matches_workspace_yaml_for_each_manipulator(manip_name: str):
    """For each manipulator listed in config/workspaces.yaml:
      - Build a single-arm env ("stack") with that manipulator.
      - Integrated manipulators use their built-in gripper.
      - Modular manipulators use gripper='robotiq'.
      - After reset(), assert env.world_aabb matches the workspace YAML.
      - Step a zero action to ensure the env runs.

    Skips cleanly if:
      - The manipulator is not registered in MANIP_REG, or
      - A modular manipulator lacks an adapter to 'robotiq'.
    """
    # Must exist in registry to build an env
    manip_cls = MANIP_REG.get(manip_name)
    if manip_cls is None:
        pytest.skip(f"Manipulator '{manip_name}' is not registered; skipping.")

    is_integrated = isinstance(manip_cls(), IntegratedManipulatorModel)

    # Build RobotOptions
    single = (
        ArmOptions(manipulator=manip_name, gripper=None)
        if is_integrated
        else ArmOptions(manipulator=manip_name, gripper="robotiq")
    )
    rob = RobotOptions(type_of_robot="single", single=single)

    # Try constructing the environment
    try:
        env = ds.make("stack", robot=rob, cameras=None, render_mode="rgb_array")
    except ValueError as e:
        msg = str(e)
        if "Adapter missing" in msg:
            pytest.skip(
                f"Manipulator '{manip_name}' has no adapter for gripper 'robotiq'; skipping.",
            )
        if "Unknown manipulator" in msg:
            pytest.skip(f"Manipulator '{manip_name}' not implemented; skipping.")
        raise

    try:
        _obs, _ = env.reset()

        # Sanity: integrated vs modular
        if is_integrated:
            assert isinstance(env.robot, IntegratedRobot), (
                f"Expected IntegratedRobot for '{manip_name}', got {type(env.robot)}"
            )
        else:
            assert isinstance(env.robot, ModularRobot), (
                f"Expected ModularRobot for '{manip_name}', got {type(env.robot)}"
            )

        # Expected AABB from YAML
        ws = _WORKSPACES[manip_name]
        exp_lo = np.asarray(ws["min"], dtype=np.float32)
        exp_hi = np.asarray(ws["max"], dtype=np.float32)

        # Base pose offset from the actual env options (default single: (0,0,0), yaw=0)
        base_pos = np.asarray(env.robot_options.layout.single.pos, dtype=np.float32)
        # Current BaseEnv computes world AABB as (lo+base_pos, hi+base_pos) for yaw=0.
        # If you later add a non-zero yaw for single layouts, update BaseEnv accordingly
        # to rotate the AABB, and adjust this expectation.
        exp_world_lo = exp_lo + base_pos
        exp_world_hi = exp_hi + base_pos

        # Actual AABB from the env
        wa = env.world_aabb
        assert hasattr(wa, "min") and hasattr(wa, "max"), "world_aabb missing"
        act_lo = _to_np(wa.min)
        act_hi = _to_np(wa.max)

        # Exact match for default single layout (yaw=0.0)
        np.testing.assert_allclose(act_lo, exp_world_lo, rtol=0, atol=1e-6)
        np.testing.assert_allclose(act_hi, exp_world_hi, rtol=0, atol=1e-6)

        # Step a zero action to ensure the env behaves
        zero_act = _zeros_like(env.action_space)
        env.step(zero_act)

    finally:
        env.close()
