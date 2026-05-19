from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero quaternion")
    # q and -q represent the same SU(2)/SO(3) rotation up to global sign.
    q = q / norm
    return q if q[0] >= 0 else -q


def su2_to_quaternion(U: np.ndarray) -> np.ndarray:
    """Convert a 2x2 SU(2) rotation matrix to q=(w,x,y,z).

    Convention:
        U = [[w - i z, -y - i x],
             [y - i x,  w + i z]].

    This matches RX/RY/RZ(theta)=exp(-i theta sigma/2).
    A global phase is removed before conversion.
    """
    U = np.asarray(U, dtype=complex).reshape(2, 2)
    det = np.linalg.det(U)
    if abs(det) == 0:
        raise ValueError("matrix determinant is zero")
    U = U / np.sqrt(det)  # remove global phase; rotations should now be in SU(2)
    w = U[0, 0].real
    z = -U[0, 0].imag
    y = U[1, 0].real
    x = -U[1, 0].imag
    return normalize_quaternion(np.array([w, x, y, z], dtype=float))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Return the SO(3) rotation matrix associated with q=(w,x,y,z)."""
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def bloch_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Return r = R(q) e_z, the Bloch vector of U(q)|0>."""
    ez = np.array([0.0, 0.0, 1.0])
    return quaternion_to_rotation_matrix(q) @ ez


def geodesic_distance_s3(q1: np.ndarray, q2: np.ndarray) -> float:
    """Projective S^3 distance; q and -q are identified."""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def quaternion_path_distances(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=float)
    return np.array([geodesic_distance_s3(qs[i], qs[i + 1]) for i in range(len(qs) - 1)])
