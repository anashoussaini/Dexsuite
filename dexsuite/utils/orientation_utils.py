"""Convert rotations using the WXYZ quaternion convention.

Torch helpers preserve input devices and use radians for roll, pitch, and yaw. Non-torch convenience wrappers create tensors on the global DexSuite device.
"""

from __future__ import annotations

import torch

from dexsuite.utils.globals import get_device


def _as_float32(x: torch.Tensor) -> torch.Tensor:
    """Return x as float32 without changing its device."""
    return x.to(dtype=torch.float32)


def quat_normalize_torch(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternions along the last dimension.

    Args:
        q: Quaternion tensor with shape (..., 4) in (w, x, y, z) order.
        eps: Small constant added to the norm for numerical stability.

    Returns:
        torch.Tensor: Normalized quaternion tensor with shape (..., 4).
    """
    q = _as_float32(q)
    return q / (q.norm(dim=-1, keepdim=True) + float(eps))


def quat_mul_wxyz_torch(q_b: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in WXYZ convention.

    Computes the Hamilton product q = q_b ⊗ q_a, which corresponds to applying
    rotation q_a first, followed by rotation q_b.

    Args:
        q_b: Quaternion(s) with shape (..., 4) in (w, x, y, z) order.
        q_a: Quaternion(s) with shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: Product quaternion(s) with shape (..., 4).
    """
    q_b, q_a = torch.broadcast_tensors(_as_float32(q_b), _as_float32(q_a))
    w_b, x_b, y_b, z_b = q_b.unbind(-1)
    w_a, x_a, y_a, z_a = q_a.unbind(-1)
    return torch.stack(
        (
            w_b * w_a - x_b * x_a - y_b * y_a - z_b * z_a,
            w_b * x_a + x_b * w_a + y_b * z_a - z_b * y_a,
            w_b * y_a - x_b * z_a + y_b * w_a + z_b * x_a,
            w_b * z_a + x_b * y_a - y_b * x_a + z_b * w_a,
        ),
        dim=-1,
    )


def rpy_to_quat_wxyz_torch(rpy: torch.Tensor) -> torch.Tensor:
    """Convert roll/pitch/yaw to a quaternion (WXYZ).

    The convention used is extrinsic X-Y-Z rotations:
        - roll about +X
        - pitch about +Y
        - yaw about +Z

    Args:
        rpy: Tensor of angles in radians with shape (..., 3) as [roll, pitch, yaw].

    Returns:
        torch.Tensor: Quaternion(s) with shape (..., 4) in (w, x, y, z) order.
    """
    rpy = _as_float32(rpy)
    half = 0.5 * rpy
    cr, cp, cy = (
        torch.cos(half[..., 0]),
        torch.cos(half[..., 1]),
        torch.cos(half[..., 2]),
    )
    sr, sp, sy = (
        torch.sin(half[..., 0]),
        torch.sin(half[..., 1]),
        torch.sin(half[..., 2]),
    )
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
        ),
        dim=-1,
    )


def yaw_quat_wxyz_torch(yaw: torch.Tensor) -> torch.Tensor:
    """Create a quaternion (WXYZ) representing a pure yaw about +Z.

    Args:
        yaw: Angle(s) in radians with shape (...) or (...,).

    Returns:
        torch.Tensor: Quaternion(s) with shape (..., 4) in (w, x, y, z) order.
    """
    yaw = _as_float32(yaw)
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_to_rpy_wxyz_torch(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (WXYZ) to roll/pitch/yaw.

    Args:
        q: Quaternion tensor with shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: RPY angles in radians with shape (..., 3) as [roll, pitch, yaw].
    """
    q = quat_normalize_torch(q)
    w, x, y, z = q.unbind(-1)
    t2 = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(t2)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: convert WXYZ quaternion to RPY on the global device.

    This mirrors the legacy helper previously exposed via dexsuite.utils.

    Args:
        quat: Quaternion(s) on any device with shape (..., 4).

    Returns:
        torch.Tensor: RPY angles in radians on get_device() with dtype float32.
    """
    return quat_to_rpy_wxyz_torch(quat.to(device=get_device(), dtype=torch.float32))


def quat_to_rotmat_wxyz_torch(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (WXYZ) to a rotation matrix.

    Args:
        q: Quaternion tensor with shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: Rotation matrix with shape (..., 3, 3).
    """
    q = quat_normalize_torch(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    r00 = ww + xx - yy - zz
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = ww - xx + yy - zz
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = ww - xx - yy + zz
    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: convert RPY to WXYZ quaternion on the global device.

    This mirrors the legacy helper previously exposed via dexsuite.utils.

    Args:
        rpy: Tensor of angles in radians with shape (..., 3).

    Returns:
        torch.Tensor: Quaternion(s) on get_device() with dtype float32.
    """
    return rpy_to_quat_wxyz_torch(rpy.to(device=get_device(), dtype=torch.float32))
