from __future__ import annotations

import numpy as np

from .quaternion_geometry import quaternion_path_distances


def principal_eigenvector(C: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    v = vecs[:, int(np.argmax(vals))]
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v / np.linalg.norm(v)


def bloch_tangents(bloch_path: np.ndarray, xs: np.ndarray | None = None) -> np.ndarray:
    R = np.asarray(bloch_path, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 3:
        raise ValueError("bloch_path must have shape (n>=3, 3)")
    if xs is None:
        return np.diff(R, axis=0)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if len(xs) != len(R):
        raise ValueError("xs and bloch_path must have the same length")
    dx = np.diff(xs)
    if np.any(dx == 0):
        raise ValueError("xs must be strictly varying")
    return np.diff(R, axis=0) / dx[:, None]


def quaternion_motion(quaternions: np.ndarray) -> np.ndarray:
    """Projective S^3 increments d(q_i,q_{i+1}).

    The distance uses |q_i.q_{i+1}|, so q and -q are identified.
    """
    Q = np.asarray(quaternions, dtype=float)
    if Q.ndim != 2 or Q.shape[1] != 4 or Q.shape[0] < 2:
        raise ValueError("quaternions must have shape (n>=2, 4)")
    return quaternion_path_distances(Q)


def readout_values(bloch_path: np.ndarray, v: np.ndarray) -> np.ndarray:
    R = np.asarray(bloch_path, dtype=float)
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    return R @ v


def readout_variations(bloch_path: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(readout_values(bloch_path, v)))


def quaternion_weighted_covariance(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> np.ndarray:
    R = np.asarray(bloch_path, dtype=float)
    Q = np.asarray(quaternions, dtype=float)
    if len(R) != len(Q):
        raise ValueError("bloch_path and quaternions must have the same length")
    D = bloch_tangents(R, xs)
    w = quaternion_motion(Q) ** power
    denom = float(np.sum(w))
    if denom <= 0:
        return D.T @ D / len(D)
    w = w / denom
    return (D * w[:, None]).T @ D


def quaternion_guided_axis(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """Return v_quat, the axis maximizing quaternion-weighted tangent energy."""
    C = quaternion_weighted_covariance(bloch_path, quaternions, xs=xs, power=power)
    return principal_eigenvector(C)


def quaternion_weighted_energy(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> float:
    D = bloch_tangents(bloch_path, xs)
    w = quaternion_motion(quaternions) ** power
    denom = float(np.sum(w))
    if denom > 0:
        w = w / denom
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    return float(np.sum(w * (D @ v) ** 2))


def hidden_motion_ratio(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    eta: float = 1e-12,
) -> float:
    """Global hidden-motion ratio sum(delta_q^2)/(sum(delta_v^2)+eta)."""
    dq = quaternion_motion(quaternions)
    dv = readout_variations(bloch_path, v)
    return float(np.sum(dq**2) / (np.sum(dv**2) + eta))


def local_hidden_motion_ratio(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    eta: float = 1e-12,
) -> np.ndarray:
    dq = quaternion_motion(quaternions)
    dv = readout_variations(bloch_path, v)
    return dq / (dv + eta)


def capture_score(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    eta: float = 1e-12,
) -> float:
    """Diagnostic score sum(delta_v^2)/(sum(delta_q^2)+eta)."""
    dq = quaternion_motion(quaternions)
    dv = readout_variations(bloch_path, v)
    return float(np.sum(dv**2) / (np.sum(dq**2) + eta))
