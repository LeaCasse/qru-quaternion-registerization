from __future__ import annotations

import numpy as np
from .bloch import PAULIS, bloch_vector, signed_coordinate


def normalize_axis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("axis v must be nonzero")
    return v / n


def projector_from_axis(v: np.ndarray) -> np.ndarray:
    """Return Pi_v^+ = (I + v.sigma)/2."""
    v = normalize_axis(v)
    observable = sum(v[i] * PAULIS[i] for i in range(3))
    return 0.5 * (np.eye(2, dtype=complex) + observable)


def probability_from_signed_coordinate(s: float) -> float:
    if s < -1 - 1e-10 or s > 1 + 1e-10:
        raise ValueError("signed coordinate must lie in [-1,1]")
    return float((1.0 + np.clip(s, -1.0, 1.0)) / 2.0)


def signed_coordinate_from_probability(p: float) -> float:
    if p < -1e-10 or p > 1 + 1e-10:
        raise ValueError("probability must lie in [0,1]")
    return float(2.0 * np.clip(p, 0.0, 1.0) - 1.0)


def projector_probability(state: np.ndarray, v: np.ndarray) -> float:
    psi = np.asarray(state, dtype=complex).reshape(2)
    psi = psi / np.linalg.norm(psi)
    Pi = projector_from_axis(v)
    p = np.vdot(psi, Pi @ psi).real
    return float(np.clip(p, 0.0, 1.0))


def ideal_ae_error_model(p: float, epsilon: float, seed: int | None = None) -> float:
    """Toy amplitude-estimation error model, not a hardware implementation."""
    rng = np.random.default_rng(seed)
    return float(np.clip(p + rng.uniform(-epsilon, epsilon), 0.0, 1.0))
