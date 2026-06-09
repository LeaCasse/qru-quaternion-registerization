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


def relative_rotation_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Relative SO(3) rotation angle in [0, pi].

    Unit quaternions q and -q are identified. The factor two converts
    projective S^3 separation into the corresponding SO(3) rotation angle.
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def quaternion_path_distances(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=float)
    return np.array([relative_rotation_angle(qs[i], qs[i + 1]) for i in range(len(qs) - 1)])


def fubini_study_distance(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Pure-state distance 2 arccos(|<psi_a|psi_b>|) in [0, pi]."""
    a = np.asarray(psi_a, dtype=complex).reshape(-1)
    b = np.asarray(psi_b, dtype=complex).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("state vectors must be non-zero")
    overlap = float(abs(np.vdot(a / na, b / nb)))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    if 1.0 - overlap < 1e-14:
        return 0.0
    return float(2.0 * np.arccos(overlap))


def bloch_geodesic_distance(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle between unit Bloch vectors in [0, pi]."""
    a = np.asarray(r_a, dtype=float).reshape(3)
    b = np.asarray(r_b, dtype=float).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Bloch vectors must be non-zero")
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    if 1.0 - dot < 1e-14:
        return 0.0
    return float(np.arccos(dot))


def quotient_quaternion_distance(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Distance on SU(2)/U(1) for states prepared from |0>.

    This equals min_phi d_rot(U_a, U_b Rz(phi)) and is evaluated through
    the equivalent Fubini--Study distance between U_a|0> and U_b|0>.
    """
    Ua = np.asarray(U_a, dtype=complex).reshape(2, 2)
    Ub = np.asarray(U_b, dtype=complex).reshape(2, 2)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    return fubini_study_distance(Ua @ ket0, Ub @ ket0)
