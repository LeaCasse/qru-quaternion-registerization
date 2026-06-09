from __future__ import annotations

import numpy as np

from .bloch import PAULIS
from .gates import ry, rz


def normalize_axis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError("axis v must be finite")
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("axis v must be nonzero")
    return v / n


def directional_observable(v: np.ndarray) -> np.ndarray:
    v = normalize_axis(v)
    return sum(v[i] * PAULIS[i] for i in range(3))


def projector_from_axis(v: np.ndarray) -> np.ndarray:
    """Return Pi_v^+ = (I + v.sigma)/2."""
    return 0.5 * (np.eye(2, dtype=complex) + directional_observable(v))


def basis_rotation_from_axis(v: np.ndarray) -> np.ndarray:
    r"""Return B_v satisfying B_v^\dagger Z B_v = v.sigma.

    For v=(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)), choose
    B_v = RY(-theta) RZ(-phi). Applying B_v before a computational-basis
    measurement therefore realizes the directional observable v.sigma.
    """
    vx, vy, vz = normalize_axis(v)
    theta = float(np.arccos(np.clip(vz, -1.0, 1.0)))
    phi = float(np.arctan2(vy, vx))
    return ry(-theta) @ rz(-phi)


def probability_from_signed_coordinate(s: float) -> float:
    if not np.isfinite(s) or s < -1 - 1e-10 or s > 1 + 1e-10:
        raise ValueError("signed coordinate must be finite and lie in [-1,1]")
    return float((1.0 + np.clip(s, -1.0, 1.0)) / 2.0)


def signed_coordinate_from_probability(p: float) -> float:
    if not np.isfinite(p) or p < -1e-10 or p > 1 + 1e-10:
        raise ValueError("probability must be finite and lie in [0,1]")
    return float(2.0 * np.clip(p, 0.0, 1.0) - 1.0)


def projector_probability(state: np.ndarray, v: np.ndarray) -> float:
    psi = np.asarray(state, dtype=complex).reshape(2)
    norm = np.linalg.norm(psi)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("state must be finite and nonzero")
    psi = psi / norm
    Pi = projector_from_axis(v)
    p = np.vdot(psi, Pi @ psi).real
    return float(np.clip(p, 0.0, 1.0))


def basis_rotation_probability(state: np.ndarray, v: np.ndarray) -> float:
    """Return the probability of |0> after applying B_v to ``state``."""
    psi = np.asarray(state, dtype=complex).reshape(2)
    psi = psi / np.linalg.norm(psi)
    rotated = basis_rotation_from_axis(v) @ psi
    return float(np.clip(abs(rotated[0]) ** 2, 0.0, 1.0))
