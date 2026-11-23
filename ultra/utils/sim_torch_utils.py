"""Pure-PyTorch math helpers used across ULTRA.

This module replaces ``isaacgym.torch_utils`` so that the code base no longer depends on Isaac Gym.  The public
names and semantics match what the ULTRA code expects from the old module:

* quaternions are stored as ``(x, y, z, w)`` (scalar last), the layout of every state tensor in ULTRA;
* ``to_torch`` defaults to ``device="cuda:0"``;
* functions are TorchScript-compatible so they can be called from ``@torch.jit.script`` code.

The module intentionally has no ``__all__`` and re-exports ``np`` / ``torch``: callers use
``from utils.sim_torch_utils import *`` and rely on those names being available.
"""

import math
from typing import Tuple

import numpy as np
import torch


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


@torch.jit.script
def quat_mul(a, b):
    """Hamilton product of two (x, y, z, w) quaternion tensors of the same shape."""
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1).view(shape)


@torch.jit.script
def quat_apply(a, b):
    """Rotate vectors ``b`` (..., 3) by quaternions ``a`` (..., 4)."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = torch.cross(xyz, b, dim=-1) * 2
    return (b + a[:, 3:] * t + torch.cross(xyz, t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    """Rotate vectors ``v`` (N, 3) by quaternions ``q`` (N, 4)."""
    q_w = q[..., -1:]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    """Rotate vectors ``v`` (N, 3) by the inverse of quaternions ``q`` (N, 4)."""
    q_w = q[..., -1:]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """Return a list of ``n_dims`` values that is ``value`` along ``axis_idx`` and ``x_value`` in slot 0."""
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    params = np.zeros((n_dims,))
    params[axis_idx] = value
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a: float, b):
    a_t = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_t) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(math.pi / 2.0, sinp), torch.asin(sinp))

    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    two_pi = 2.0 * math.pi
    return roll % two_pi, pitch % two_pi, yaw % two_pi


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: Tuple[int, int], device: str):
    return (upper - lower) * torch.rand(shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape: Tuple[int, int], device: str):
    angle = torch_rand_float(-math.pi, math.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
