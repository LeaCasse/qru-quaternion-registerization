from __future__ import annotations

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = np.stack([X, Y, Z])


def normalize_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).reshape(2)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("zero vector is not a quantum state")
    return state / norm


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Return (<X>, <Y>, <Z>) for a normalized single-qubit pure state."""
    psi = normalize_state(state)
    return np.array([np.vdot(psi, P @ psi).real for P in PAULIS], dtype=float)


def signed_coordinate(state_or_bloch: np.ndarray, v: np.ndarray) -> float:
    """Return s_v = v.r for either a state vector or a Bloch vector."""
    arr = np.asarray(state_or_bloch)
    r = bloch_vector(arr) if arr.shape == (2,) else np.asarray(arr, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    nv = np.linalg.norm(v)
    if nv == 0:
        raise ValueError("axis v must be nonzero")
    v = v / nv
    return float(np.dot(v, r))
