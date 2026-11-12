"""Define left and right Sharpa hand gripper models.

Per-joint tables are keyed by joint name (without the ``right_``/``left_`` prefix)
and materialised in Genesis DoF order. Genesis lays out the DoFs of an MJCF model
in breadth-first order over the body tree, not in the depth-first document order
of the XML, so the tensors are built from the model file rather than typed in
XML order.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from typing import ClassVar

import torch

from dexsuite.core.components import GripperModel
from dexsuite.core.registry import register_gripper
from dexsuite.utils.models_utils import package_asset_path

# joint name (side prefix stripped) -> (right-hand home q, q_lo, q_hi, kp, kv); radians / N*m/rad / N*m*s/rad
# kp ~ 2500 * (armature + link inertia about the joint) -> omega_n ~ 50 rad/s; kv = 1.8 * sqrt(kp * inertia) (zeta 0.9);
# pinky_CMC kp is capped by the explicit-PD margin kp * dt^2 / inertia <= 0.5 at the 0.01 s physics step
_JOINT_TABLE: dict[str, tuple[float, float, float, float, float]] = {
    "thumb_CMC_FE": (0.0, -0.1745, 1.9199, 9.1, 0.328),
    "thumb_CMC_AA": (-0.12, -0.3491, 0.3491, 13.2, 0.403),
    "thumb_MCP_FE": (0.0, -0.5236, 1.3963, 6.7, 0.242),
    "thumb_MCP_AA": (-0.08, -0.3491, 0.3491, 6.7, 0.242),
    "thumb_IP": (0.0, 0.0, 1.7453, 1.5, 0.054),
    "index_MCP_FE": (0.0, -0.17453293, 1.5708, 7, 0.252),
    "index_MCP_AA": (0.08, -0.3491, 0.3491, 7, 0.252),
    "index_PIP": (0.0, 0.0, 1.7453, 1.5, 0.055),
    "index_DIP": (0.0, 0.0, 1.3963, 1.05, 0.038),
    "middle_MCP_FE": (0.0, -0.17453293, 1.5708, 7, 0.252),
    "middle_MCP_AA": (0.0, -0.3491, 0.3491, 7, 0.252),
    "middle_PIP": (0.0, 0.0, 1.7453, 1.5, 0.055),
    "middle_DIP": (0.0, 0.0, 1.3963, 1.05, 0.038),
    "ring_MCP_FE": (0.0, -0.17453293, 1.5708, 7, 0.252),
    "ring_MCP_AA": (-0.06, -0.3491, 0.3491, 7, 0.252),
    "ring_PIP": (0.0, 0.0, 1.7453, 1.5, 0.055),
    "ring_DIP": (0.0, 0.0, 1.3963, 1.05, 0.038),
    "pinky_CMC": (0.0, 0.0, 0.2618, 0.8, 0.021),
    "pinky_MCP_FE": (0.0, -0.17453293, 1.5708, 7, 0.252),
    "pinky_MCP_AA": (-0.08, -0.3491, 0.3491, 7, 0.252),
    "pinky_PIP": (0.0, 0.0, 1.7453, 1.5, 0.055),
    "pinky_DIP": (0.0, 0.0, 1.3963, 1.05, 0.038),
}


def genesis_dof_joint_names(model_path: str) -> list[str]:
    """Return the MJCF joint names in Genesis DoF order (breadth-first over bodies)."""
    worldbody = ET.parse(model_path).getroot().find("worldbody")
    queue = deque(worldbody.findall("body"))
    names: list[str] = []
    while queue:
        body = queue.popleft()
        names.extend(joint.get("name") for joint in body.findall("joint"))
        queue.extend(body.findall("body"))
    return names


def _joint_tables(model_path: str, side: str) -> tuple[torch.Tensor, ...]:
    names = genesis_dof_joint_names(model_path)
    keys = [name.removeprefix(f"{side}_") for name in names]
    if set(keys) != set(_JOINT_TABLE) or len(keys) != len(_JOINT_TABLE):
        raise ValueError(
            f"Sharpa joint set in {model_path} does not match the table: {keys}",
        )
    rows = [_JOINT_TABLE[key] for key in keys]
    home_sign = 1.0 if side == "right" else -1.0
    columns = zip(*rows)
    home, q_lo, q_hi, kp, kv = (torch.tensor(col, dtype=torch.float32) for col in columns)
    return home * home_sign, q_lo, q_hi, kp, kv


class _SharpaBase(GripperModel):
    """Provide shared Sharpa hand settings."""

    dof: int = 22
    control_dofs_index: ClassVar[list[int]] = list(range(dof))
    batch_fixed_verts: bool = True

    force_range: torch.Tensor = torch.full((dof,), 20.0, dtype=torch.float32)

    tcp_pose: torch.Tensor = torch.tensor(
        [0.0, 0.0, 0.16, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )


@register_gripper("sharpa_right")
@register_gripper("sharpa")
class SharpaRight(_SharpaBase):
    """Represent the right Sharpa hand."""

    model_path: str = package_asset_path(
        __package__, "right_sharpa_wave_with_flange.xml"
    )
    root_link: str = "right_hand_C_MC"
    home_q, q_lo, q_hi, kp, kv = _joint_tables(model_path, "right")


@register_gripper("sharpa_fardexbot")
class SharpaFardexbotRight(_SharpaBase):
    """Represent the right Sharpa hand carrying the fardexbox camera mount plate and the D405 body.

    This is the hand the scripted Sharpa datasets were recorded with: the plain right hand plus the wrist-camera
    mount, which adds 0.5 kg to the root link and collides with the scene.
    """

    model_path: str = package_asset_path(__package__, "right_sharpa_fardexbot.xml")
    root_link: str = "right_hand_C_MC"
    home_q, q_lo, q_hi, kp, kv = _joint_tables(model_path, "right")


@register_gripper("sharpa_left")
class SharpaLeft(_SharpaBase):
    """Represent the left Sharpa hand."""

    model_path: str = package_asset_path(
        __package__, "left_sharpa_wave_with_flange.xml"
    )
    root_link: str = "left_hand_C_MC"
    home_q, q_lo, q_hi, kp, kv = _joint_tables(model_path, "left")
