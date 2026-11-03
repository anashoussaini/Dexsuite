"""OpenVR helpers for Vive controllers and trackers.

This module is adapted from common OpenVR "triad" utilities and provides helper
classes and math utilities for reading poses from SteamVR devices.

The openvr dependency is optional. Importing this module succeeds even when
openvr is not installed; however, attempting to instantiate triad_openvr or
call OpenVR-specific functions will raise a RuntimeError with instructions.
"""

from __future__ import annotations

import json
import math
import sys
import time
from functools import cache

import numpy as np

try:
    import openvr  # type: ignore
except Exception as _e:  # pragma: no cover
    openvr = None  # type: ignore[assignment]
    _OPENVR_IMPORT_ERROR: Exception | None = _e
else:  # pragma: no cover
    _OPENVR_IMPORT_ERROR = None


def _require_openvr():  # pragma: no cover
    if openvr is None:
        raise RuntimeError(
            "Vive device support requires 'openvr' (pip install openvr) and a working "
            "SteamVR installation.",
        ) from _OPENVR_IMPORT_ERROR
    return openvr


class TriadLocalPose:
    """Helper utilities for working with local poses and rotations.

    This class contains only static helper methods for converting between
    rotation representations (matrices, Euler angles, quaternions),
    computing distances/angles between rotations, and extracting pose
    information from devices that expose different pose formats.

    All methods are @staticmethod and operate on numpy arrays or numeric
    scalars. No instance state is stored.
    """

    @staticmethod
    def orthonormalize(R: np.ndarray) -> np.ndarray:
        """Project a 3x3 matrix to the nearest proper rotation matrix.

        This uses Singular Value Decomposition (SVD) to compute the closest
        orthonormal matrix with determinant +1 (a proper rotation).

        Args:
            R: A 3x3 (or shape-compatible) array representing a rotation-like
                matrix.

        Returns:
            A 3x3 numpy.ndarray representing a proper rotation matrix.
        """
        U, S, Vt = np.linalg.svd(R)
        R_hat = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        return R_hat

    @staticmethod
    def euler_rpy_from_R_zyx(R: np.ndarray):
        """Extract roll, pitch, yaw from a rotation matrix (ZYX order).

        The function assumes the intrinsic ZYX Tait-Bryan rotation sequence
        and returns the angles in radians in the order (roll, pitch, yaw).

        Args:
            R: A 3x3 rotation matrix.

        Returns:
            A tuple (roll, pitch, yaw) in radians.
        """
        v = -R[2, 0]
        v = float(np.clip(v, -1.0, 1.0))
        pitch = math.asin(v)
        if abs(v) < 0.999999:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = 0.0
            yaw = math.atan2(-R[0, 1], R[1, 1])
        return roll, pitch, yaw

    @staticmethod
    def R_from_euler_zyx(roll, pitch, yaw):
        """Build a 3x3 rotation matrix from ZYX (roll, pitch, yaw) angles.

        Args:
            roll: Rotation about X axis in radians.
            pitch: Rotation about Y axis in radians.
            yaw: Rotation about Z axis in radians.

        Returns:
            A 3x3 numpy.ndarray representing the rotation.
        """
        cx, sx = math.cos(roll), math.sin(roll)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cz, sz = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return Rz @ Ry @ Rx

    @staticmethod
    def R_from_quat_wxyz(w, x, y, z):
        """Convert a quaternion (w, x, y, z) into a 3x3 rotation matrix.

        The quaternion will be normalized before conversion; if the quaternion
        has zero norm the identity matrix is returned.

        Args:
            w, x, y, z: Quaternion components.

        Returns:
            A 3x3 numpy.ndarray rotation matrix.
        """
        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n == 0.0:
            return np.eye(3)
        w, x, y, z = w / n, x / n, y / n, z / n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )
        return R

    @staticmethod
    def quat_wxyz_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
        """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z).

        The algorithm is numerically stable and handles different diagonal
        dominance cases. The returned quaternion is normalized and the scalar
        part w is made non-negative to reduce sign ambiguity.

        Args:
            R: A 3x3 rotation matrix (array-like).

        Returns:
            A tuple (w, x, y, z) of floats representing the quaternion.
        """
        # Ensure 3x3
        R = np.asarray(R, dtype=float)
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        tr = m00 + m11 + m22

        if tr > 0.0:
            S = math.sqrt(max(tr + 1.0, 0.0)) * 2.0
            w = 0.25 * S
            x = (m21 - m12) / S
            y = (m02 - m20) / S
            z = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(max(1.0 + m00 - m11 - m22, 0.0)) * 2.0
            w = (m21 - m12) / S
            x = 0.25 * S
            y = (m01 + m10) / S
            z = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(max(1.0 - m00 + m11 - m22, 0.0)) * 2.0
            w = (m02 - m20) / S
            x = (m01 + m10) / S
            y = 0.25 * S
            z = (m12 + m21) / S
        else:
            S = math.sqrt(max(1.0 - m00 - m11 + m22, 0.0)) * 2.0
            w = (m10 - m01) / S
            x = (m02 + m20) / S
            y = (m12 + m21) / S
            z = 0.25 * S

        # Normalize and make sign consistent
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm > 0:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
        if w < 0.0:  # optional: keep w >= 0 to avoid sign flips
            w, x, y, z = -w, -x, -y, -z

        return float(w), float(x), float(y), float(z)

    @staticmethod
    def _angle_between_R(Ra: np.ndarray, Rb: np.ndarray) -> float:
        """Compute the smallest rotation angle between two rotation matrices.

        The angle returned is in radians and lies in [0, pi]. It is computed
        from the relative rotation Ra.T @ Rb using the trace formula.

        Args:
            Ra: A 3x3 rotation matrix.
            Rb: A 3x3 rotation matrix.

        Returns:
            The rotation angle (float, radians) between Ra and Rb.
        """
        Rrel = Ra.T @ Rb
        c = (np.trace(Rrel) - 1.0) * 0.5
        c = float(np.clip(c, -1.0, 1.0))
        return math.acos(c)

    @staticmethod
    def try_matrix(dev):
        """Try to obtain a device pose as (R, p) from a 3x4 matrix.

        The function checks whether the given device exposes a
        get_pose_matrix method. If present and returns a valid 3x4 matrix
        (or flat array of length 12), the rotation matrix R (3x3) and
        translation vector p (3,) are returned.

        Args:
            dev: Device-like object that may implement get_pose_matrix().

        Returns:
            A tuple (R, p) where R is a 3x3 numpy array and p is a length-3
            numpy array, or None if the pose is not available or invalid.
        """
        if not hasattr(dev, "get_pose_matrix"):
            return None
        M = dev.get_pose_matrix()
        if M is None:
            return None
        M = np.asarray(M, dtype=float)
        if M.size == 12 and M.shape != (3, 4):
            M = M.reshape(3, 4)
        if M.shape == (3, 4):
            R = M[:, :3]
            p = M[:, 3]
            return R, p
        return None

    @staticmethod
    def try_quat(dev):
        """Try to obtain a device pose encoded as a quaternion and position.

        The function looks for a get_pose_quaternion() method on the device.
        Depending on the returned layout it will extract position and
        quaternion components and convert the quaternion to a rotation
        matrix.

        Args:
            dev: Device-like object that may implement get_pose_quaternion().

        Returns:
            A tuple (R, p) where R is a 3x3 numpy array and p is a length-3
            numpy array, or None if the pose is not available or malformed.
        """
        if not hasattr(dev, "get_pose_quaternion"):
            return None
        q = dev.get_pose_quaternion()
        if q is None:
            return None
        q = list(q)

        if len(q) >= 7:
            first_three = q[0:3]
            rest = q[3:7]
            if all(np.isfinite(first_three)) and all(np.isfinite(rest)):
                x, y, z = (
                    float(first_three[0]),
                    float(first_three[1]),
                    float(first_three[2]),
                )
                w, qx, qy, qz = (
                    float(rest[0]),
                    float(rest[1]),
                    float(rest[2]),
                    float(rest[3]),
                )
                R = TriadLocalPose.R_from_quat_wxyz(w, qx, qy, qz)
                p = np.array([x, y, z], dtype=float)
                return R, p
        if len(q) == 4:
            return None
        return None

    @staticmethod
    def try_euler(dev):
        """Try to obtain a device pose provided as Euler angles plus position.

        The function expects get_pose_euler() to return an iterable with at
        least six finite numbers: x, y, z, yaw, pitch, roll (degrees). Yaw,
        pitch and roll are converted to radians and composed into a rotation
        matrix using ZYX order.

        Args:
            dev: Device-like object that may implement get_pose_euler().

        Returns:
            A tuple (R, p) where R is a 3x3 numpy array and p is a length-3
            numpy array, or None if the pose is not available or malformed.
        """
        if not hasattr(dev, "get_pose_euler"):
            return None
        e = dev.get_pose_euler()
        if e is None:
            return None
        e = list(e)
        if len(e) >= 6 and all(np.isfinite(ei) for ei in e[:6]):
            x, y, z = float(e[0]), float(e[1]), float(e[2])
            yaw, pitch, roll = map(float, e[3:6])
            roll_r, pitch_r, yaw_r = (
                math.radians(roll),
                math.radians(pitch),
                math.radians(yaw),
            )
            R = TriadLocalPose.R_from_euler_zyx(roll_r, pitch_r, yaw_r)
            p = np.array([x, y, z], dtype=float)
            return R, p
        return None

    @staticmethod
    def read_pose(dev, source="auto"):
        """Read a device pose using the requested source format.

        Args:
            dev: Device-like object offering one or more pose query methods.
            source: One of {"matrix", "quat", "euler", "auto"}. When
                "auto" the function will prefer quaternion-based reading and
                fall back where appropriate.

        Returns:
            A tuple (R, p) where R is 3x3 numpy array and p is length-3
            numpy array, or None if no valid pose could be read.
        """
        if source == "matrix":
            return TriadLocalPose.try_matrix(dev)
        if source == "quat":
            return TriadLocalPose.try_quat(dev)
        if source == "euler":
            return TriadLocalPose.try_euler(dev)
        # Default fallback
        return TriadLocalPose.try_quat(dev)


# Function to print out text but instead of starting a new line it will overwrite the existing line
def update_text(txt):
    """Write text to stdout by overwriting the current line.

    This is a convenience helper for showing a progress-like message on a
    single terminal line.

    Args:
        txt: Text to display (string).
    """
    sys.stdout.write("\r" + txt)
    sys.stdout.flush()


# Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Euler angles (in degrees)
def convert_to_euler(pose_mat):
    """Convert a 3x4 pose matrix to position and Euler angles (degrees).

    The input is assumed to be a 3x4 matrix where the left 3x3 block is a
    rotation matrix and the rightmost column is a translation vector. The
    returned Euler angles follow a yaw/pitch/roll convention and are
    expressed in degrees.

    Args:
        pose_mat: 3x4 array-like pose matrix.

    Returns:
        A list [x, y, z, yaw, pitch, roll] where position is in the same
        units as the input matrix and angles are in degrees.
    """
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, yaw, pitch, roll]


# Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Quaternion
def convert_to_quaternion(pose_mat):
    """Convert a 3x4 pose matrix to position and quaternion (w, x, y, z).

    Uses the conventional conversion from a rotation matrix to a quaternion
    and extracts the translation from the rightmost column of the 3x4
    matrix. The algorithm guards the square-root input with abs() to avoid
    spurious complex results due to rounding.

    Args:
        pose_mat: 3x4 array-like pose matrix.

    Returns:
        A list [x, y, z, r_w, r_x, r_y, r_z] where (r_w, r_x, r_y, r_z)
        is the quaternion and x,y,z the translation.
    """
    # Per issue #2, adding a abs() so that sqrt only results in real numbers
    # Adding a small epsilon to avoid zero division errors
    epsilon = 1e-8

    r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
    r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * (r_w + epsilon))
    r_y = (pose_mat[0][2] - pose_mat[2][0]) / (4 * (r_w + epsilon))
    r_z = (pose_mat[1][0] - pose_mat[0][1]) / (4 * (r_w + epsilon))

    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, r_w, r_x, r_y, r_z]


# def convert_to_quaternion(pose_mat):
#     """Stable conversion of a 3x4 pose matrix to [x,y,z,qw,qx,qy,qz]."""
#     M = np.asarray(pose_mat, dtype=float)
#     if M.size == 12 and M.shape != (3, 4):
#         M = M.reshape(3, 4)
#     R = M[:, :3]
#     t = M[:, 3]
#     # Make sure R is a proper rotation before extracting the quaternion
#     R = TriadLocalPose.orthonormalize(R)
#     qw, qx, qy, qz = TriadLocalPose.quat_wxyz_from_R(R)
#     return [float(t[0]), float(t[1]), float(t[2]), float(qw), float(qx), float(qy), float(qz)]


# Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer:
    """Buffer for accumulating pose samples and converted representations.

    The buffer stores timestamps, Cartesian positions, Euler angles (deg),
    and quaternion components for each appended 3x4 pose matrix. It is a
    minimal container useful for plotting or lightweight logging.
    """

    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self, pose_mat, t):
        """Append a pose matrix sample to the buffer.

        The provided pose matrix is expected to be a 3x4 array-like object. A
        conversion to Euler angles (degrees) and quaternion components is
        performed and stored alongside the supplied timestamp.

        Args:
            pose_mat: 3x4 array-like pose matrix.
            t: Timestamp (float) associated with the sample.
        """
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] / pose_mat[0][0]))
        self.pitch.append(
            180
            / math.pi
            * math.atan(
                -1
                * pose_mat[2][0]
                / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2)),
            ),
        )
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] / pose_mat[2][2]))
        r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w))
        self.r_y.append((pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w))
        self.r_z.append((pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w))


def get_pose(vr_obj):
    """Query OpenVR for the current device-to-absolute tracking poses.

    Args:
        vr_obj: The OpenVR system/VR interface instance (typically returned
            by openvr.init()).

    Returns:
        The list/array returned by OpenVR containing pose objects for all
        tracked devices.
    """
    ov = _require_openvr()
    return vr_obj.getDeviceToAbsoluteTrackingPose(
        ov.TrackingUniverseStanding,
        0,
        ov.k_unMaxTrackedDeviceCount,
    )


class vr_tracked_device:
    """Representation of a tracked OpenVR device with convenience helpers.

    This lightweight wrapper stores the OpenVR instance, the tracked device
    index and its high-level device class. It offers methods to query
    serial/model/battery properties and to read current pose/state in a
    few common formats.
    """

    def __init__(self, vr_obj, index, device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj

    @cache
    def get_serial(self):
        """Return the serial number string for this tracked device.

        The result is cached to avoid repeated property lookups.

        Returns:
            A bytes or str object as returned by OpenVR for the serial.
        """
        return self.vr.getStringTrackedDeviceProperty(
            self.index,
            openvr.Prop_SerialNumber_String,
        )

    def get_model(self):
        """Return the model number/name property for this tracked device.

        Returns:
            A bytes or str object representing the model identifier.
        """
        return self.vr.getStringTrackedDeviceProperty(
            self.index,
            openvr.Prop_ModelNumber_String,
        )

    def get_battery_percent(self):
        """Return the device battery percentage as a float in [0.0, 1.0].

        Returns:
            A float indicating battery percentage (OpenVR may use 0-1).
        """
        return self.vr.getFloatTrackedDeviceProperty(
            self.index,
            openvr.Prop_DeviceBatteryPercentage_Float,
        )

    def is_charging(self):
        """Return whether the device is currently charging (boolean).

        Returns:
            True if charging according to OpenVR, otherwise False.
        """
        return self.vr.getBoolTrackedDeviceProperty(
            self.index,
            openvr.Prop_DeviceIsCharging_Bool,
        )

    def sample(self, num_samples, sample_rate):
        """Take a timed sequence of pose samples from this tracked device.

        Args:
            num_samples: Number of samples to collect (int).
            sample_rate: Sampling rate in Hz (float).

        Returns:
            A pose_sample_buffer containing the collected samples and converted
            representations.
        """
        interval = 1 / sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(
                pose[self.index].mDeviceToAbsoluteTracking,
                time.time() - sample_start,
            )
            sleep_time = interval - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        return rtn

    def get_pose_euler(self, pose=None):
        """Return the device pose as Euler angles and position (or None).

        If a pose object is provided it will be used, otherwise a fresh
        query is made. When the tracked pose is valid a list is returned in
        the format produced by convert_to_euler, otherwise None.

        Args:
            pose: Optional pose snapshot (as returned by get_pose).

        Returns:
            A list [x,y,z,yaw,pitch,roll] or None if the pose is invalid.
        """
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_pose_matrix(self, pose=None):
        """Return the raw 3x4 device pose matrix when available, else None.

        Args:
            pose: Optional pose snapshot (as returned by get_pose).

        Returns:
            The 3x4 matrix (array-like) or None if the pose is invalid.
        """
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None

    def get_velocity(self, pose=None):
        """Return the linear velocity vector for the device if available.

        Args:
            pose: Optional pose snapshot (as returned by get_pose).

        Returns:
            The velocity vector or None if not available.
        """
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vVelocity
        else:
            return None

    def get_angular_velocity(self, pose=None):
        """Return the angular velocity vector for the device if available.

        Args:
            pose: Optional pose snapshot (as returned by get_pose).

        Returns:
            The angular velocity vector or None if not available.
        """
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vAngularVelocity
        else:
            return None

    def get_pose_quaternion(self, pose=None):
        """Return the device pose as a quaternion + position, or None.

        If the device pose is valid the quaternion is returned in the same
        layout produced by convert_to_quaternion, otherwise None.

        Args:
            pose: Optional pose snapshot (as returned by get_pose).

        Returns:
            A list [x,y,z,r_w,r_x,r_y,r_z] or None.
        """
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def controller_state_to_dict(self, pControllerState):
        """Convert an OpenVR controller state struct into a friendly dict.

        The returned dictionary exposes useful fields (trigger value, trackpad
        axes, button pressed/touched masks and convenience booleans) suitable
        for higher-level handling of inputs.

        Note: Implementation borrowed from an external gist and follows the
        OpenVR controller state layout.

        Args:
            pControllerState: The raw controller state struct returned by
                OpenVR.

        Returns:
            A dict with human-friendly keys for common controller inputs.
        """
        # This function is graciously borrowed from https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
        # docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
        d = {}
        d["unPacketNum"] = pControllerState.unPacketNum
        # on trigger .y is always 0.0 says the docs
        d["trigger"] = pControllerState.rAxis[1].x
        # 0.0 on trigger is fully released
        # -1.0 to 1.0 on joystick and trackpads
        d["trackpad_x"] = pControllerState.rAxis[0].x
        d["trackpad_y"] = pControllerState.rAxis[0].y
        # These are published and always 0.0
        # for i in range(2, 5):
        #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
        #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
        d["ulButtonPressed"] = pControllerState.ulButtonPressed
        d["ulButtonTouched"] = pControllerState.ulButtonTouched
        # To make easier to understand what is going on
        # Second bit marks menu button
        d["menu_button"] = bool(pControllerState.ulButtonPressed >> 1 & 1)
        # 32 bit marks trackpad
        d["trackpad_pressed"] = bool(pControllerState.ulButtonPressed >> 32 & 1)
        d["trackpad_touched"] = bool(pControllerState.ulButtonTouched >> 32 & 1)
        # third bit marks grip button
        d["grip_button"] = bool(pControllerState.ulButtonPressed >> 2 & 1)
        # System button can't be read, if you press it
        # the controllers stop reporting
        return d

    def get_controller_inputs(self):
        """Poll the OpenVR controller state and return a parsed dict.

        Returns:
            A dict as produced by controller_state_to_dict describing the
            current controller inputs.
        """
        result, state = self.vr.getControllerState(self.index)
        return self.controller_state_to_dict(state)

    def trigger_haptic_pulse(self, duration_micros=1000, axis_id=0):
        """Cause a short haptic pulse on supported devices.

        Args:
            duration_micros: Pulse duration in microseconds (int).
            axis_id: Haptic axis/index for the device (int).
        """
        self.vr.triggerHapticPulse(self.index, axis_id, duration_micros)


class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        """Return the mode label of a tracking reference (decoded string).

        The returned string is decoded from bytes to UTF-8 and uppercased for
        convenience.
        """
        return (
            self.vr.getStringTrackedDeviceProperty(
                self.index,
                openvr.Prop_ModeLabel_String,
            )
            .decode("utf-8")
            .upper()
        )

    def sample(self, num_samples, sample_rate):
        """Notify that sampling a static tracking reference is unnecessary.

        Tracking references (base stations) typically do not move; this
        method prints a warning to indicate sampling is of limited use.
        """
        print("Warning: Tracking References do not move, sample isn't much use...")


class triad_openvr:
    def __init__(self, configfile_path=None):
        """Initialize an OpenVR helper instance and discover tracked devices.

        When a JSON configfile_path is provided it will be used to map known
        device serial numbers to friendly names and types. Otherwise devices
        are discovered automatically and assigned generated names.

        Args:
            configfile_path: Optional path to a JSON config describing known
                devices (defaults to None).
        """
        ov = _require_openvr()
        # Initialize OpenVR in the
        self.vr = ov.init(ov.VRApplication_Other)
        self.vrsystem = ov.VRSystem()

        # Initializing object to hold indexes for various tracked objects
        self.object_names = {
            "Tracking Reference": [],
            "HMD": [],
            "Controller": [],
            "Tracker": [],
        }
        self.devices = {}
        self.device_index_map = {}
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            ov.TrackingUniverseStanding,
            0,
            ov.k_unMaxTrackedDeviceCount,
        )

        # Loading config file
        if configfile_path:
            try:
                with open(configfile_path) as json_data:
                    config = json.load(json_data)
            except (
                OSError
            ):  # parent of IOError, OSError *and* WindowsError where available
                print("config.json not found.")
                exit(1)

            # Iterate through the pose list to find the active devices and determine their type
            for i in range(ov.k_unMaxTrackedDeviceCount):
                if poses[i].bDeviceIsConnected:
                    device_serial = self.vr.getStringTrackedDeviceProperty(
                        i,
                        ov.Prop_SerialNumber_String,
                    ).decode("utf-8")
                    for device in config["devices"]:
                        if device_serial == device["serial"]:
                            device_name = device["name"]
                            self.object_names[device["type"]].append(device_name)
                            self.devices[device_name] = vr_tracked_device(
                                self.vr,
                                i,
                                device["type"],
                            )
        else:
            # Iterate through the pose list to find the active devices and determine their type
            for i in range(ov.k_unMaxTrackedDeviceCount):
                if poses[i].bDeviceIsConnected:
                    self.add_tracked_device(i)

    def __del__(self):
        """Shutdown the OpenVR runtime when the helper is being destroyed."""
        if openvr is None:
            return
        openvr.shutdown()

    def get_pose(self):
        """Return the latest OpenVR device-to-absolute tracking poses.

        This is a thin wrapper around the module-level get_pose helper that
        forwards this instance's OpenVR interface.
        """
        return get_pose(self.vr)

    def poll_vr_events(self):
        """Poll OpenVR events and update tracked device discovery.

        The method iterates over pending OpenVR events and handles device
        activation and deactivation events by adding or removing the
        corresponding tracked device wrappers.
        """
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                self.add_tracked_device(event.trackedDeviceIndex)
            elif (
                event.eventType == openvr.VREvent_TrackedDeviceDeactivated
                and event.trackedDeviceIndex in self.device_index_map
            ):
                # If we were already tracking this device, quit tracking it.
                self.remove_tracked_device(event.trackedDeviceIndex)

    def add_tracked_device(self, tracked_device_index):
        """Add a tracked device by index and create a helper wrapper.

        The created wrapper is stored in self.devices and a generated friendly
        name is appended to self.object_names under the
        appropriate device class.

        Args:
            tracked_device_index: The OpenVR tracked device index (int).
        """
        i = tracked_device_index
        device_class = self.vr.getTrackedDeviceClass(i)
        if device_class == openvr.TrackedDeviceClass_Controller:
            device_name = "controller_" + str(len(self.object_names["Controller"]) + 1)
            self.object_names["Controller"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "Controller")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_HMD:
            device_name = "hmd_" + str(len(self.object_names["HMD"]) + 1)
            self.object_names["HMD"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "HMD")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_GenericTracker:
            device_name = "tracker_" + str(len(self.object_names["Tracker"]) + 1)
            self.object_names["Tracker"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "Tracker")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_TrackingReference:
            device_name = "tracking_reference_" + str(
                len(self.object_names["Tracking Reference"]) + 1,
            )
            self.object_names["Tracking Reference"].append(device_name)
            self.devices[device_name] = vr_tracking_reference(
                self.vr,
                i,
                "Tracking Reference",
            )
            self.device_index_map[i] = device_name

    def remove_tracked_device(self, tracked_device_index):
        """Remove a tracked device previously discovered by index.

        If the index is not known an Exception is raised.

        Args:
            tracked_device_index: The OpenVR tracked device index (int).
        """
        if tracked_device_index in self.device_index_map:
            device_name = self.device_index_map[tracked_device_index]
            self.object_names[self.devices[device_name].device_class].remove(
                device_name,
            )
            del self.device_index_map[tracked_device_index]
            del self.devices[device_name]
        else:
            raise Exception(
                f"Tracked device index {tracked_device_index} not valid. Not removing.",
            )

    def rename_device(self, old_device_name, new_device_name):
        """Rename a discovered device in the internal registries.

        This updates both the self.devices mapping and the corresponding entry
        in self.object_names for the device's class.

        Args:
            old_device_name: Existing device key/name (str).
            new_device_name: New name to assign (str).
        """
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        for i in range(
            len(self.object_names[self.devices[new_device_name].device_class]),
        ):
            if (
                self.object_names[self.devices[new_device_name].device_class][i]
                == old_device_name
            ):
                self.object_names[self.devices[new_device_name].device_class][i] = (
                    new_device_name
                )

    def print_discovered_objects(self):
        """Print a summary of discovered tracked devices and their properties.

        For tracking references additional mode/model information is printed.
        """
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type]) != 1:
                plural += "s"
            print("Found " + str(len(self.object_names[device_type])) + " " + plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print(
                        "  "
                        + device
                        + " ("
                        + self.devices[device].get_serial()
                        + ", Mode "
                        + self.devices[device].get_model()
                        + ", "
                        + self.devices[device].get_model()
                        + ")",
                    )
                else:
                    print(
                        "  "
                        + device
                        + " ("
                        + self.devices[device].get_serial()
                        + ", "
                        + self.devices[device].get_model()
                        + ")",
                    )
