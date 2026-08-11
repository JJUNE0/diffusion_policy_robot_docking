"""Minimal SE(2) utilities.

Conventions (kept consistent with the repo's body-frame convention, see
``utils/preprocessing.py``):

* A pose is an ``np.ndarray`` of shape ``(3,)`` -> ``[x, y, theta]``.
* ``+x`` is robot **forward**, ``+y`` is robot **left**, ``theta`` is CCW (rad).
* ``pose`` represents the rigid transform that maps a point expressed in the
  *child* frame into the *parent* frame:  ``p_parent = R(theta) @ p_child + t``.

These helpers are deliberately tiny and dependency-free (numpy only) so the
evaluation tooling stays decoupled from the learned-policy / torch stack.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def matrix(pose: np.ndarray) -> np.ndarray:
    """Return the 3x3 homogeneous matrix of an ``[x, y, theta]`` pose."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [[c, -s, x],
         [s, c, y],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def from_matrix(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`matrix` -> ``[x, y, theta]``."""
    return np.array([m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])], dtype=np.float64)


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose composition ``a ∘ b`` (apply ``b`` then ``a``)."""
    return from_matrix(matrix(a) @ matrix(b))


def inverse(pose: np.ndarray) -> np.ndarray:
    """Inverse pose such that ``compose(pose, inverse(pose)) == identity``."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    # R^T, -R^T t
    xi = -(c * x + s * y)
    yi = -(-s * x + c * y)
    return np.array([xi, yi, wrap_angle(-th)], dtype=np.float64)


def transform_points(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply ``pose`` to an ``(N, 2)`` array of points (child -> parent frame)."""
    pts = np.asarray(pts, dtype=np.float64)
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ R.T + np.array([x, y], dtype=np.float64)


def relative_pose(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pose of ``target`` expressed in the frame of ``reference``."""
    return compose(inverse(reference), target)


def pose_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return ``(translation_dist, |angle_diff|)`` between two poses."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.hypot(d[0], d[1])), float(abs(wrap_angle(d[2])))
