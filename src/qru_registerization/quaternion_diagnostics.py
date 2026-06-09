from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion_geometry import fubini_study_distance, quaternion_path_distances


@dataclass(frozen=True)
class AxisSelectionResult:
    axis: np.ndarray | None
    eigenvalues: np.ndarray
    eigengap: float
    identifiable: bool
    energy: float
    method: str


def _validated_grid(n: int, xs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if xs is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(xs, dtype=float).reshape(-1)
        if len(x) != n:
            raise ValueError("xs and trajectory must have the same length")
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("xs must be strictly increasing")
    return x, dx


def _canonical_axis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("axis must be non-zero")
    v = v / norm
    idx = int(np.argmax(np.abs(v)))
    return -v if v[idx] < 0 else v


def principal_axis_result(
    C: np.ndarray,
    *,
    method: str,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelectionResult:
    C = np.asarray(C, dtype=float).reshape(3, 3)
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    leading = float(max(vals[0], 0.0))
    eigengap = float((vals[0] - vals[1]) / (abs(vals[0]) + energy_tol))
    identifiable = bool(leading > energy_tol and eigengap > eigengap_tol)
    axis = _canonical_axis(vecs[:, 0]) if identifiable else None
    return AxisSelectionResult(
        axis=axis,
        eigenvalues=vals,
        eigengap=eigengap,
        identifiable=identifiable,
        energy=leading,
        method=method,
    )


def principal_eigenvector(C: np.ndarray) -> np.ndarray:
    """Backward-compatible principal eigenvector without identifiability rejection."""
    C = np.asarray(C, dtype=float).reshape(3, 3)
    vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
    return _canonical_axis(vecs[:, int(np.argmax(vals))])


def bloch_tangents(bloch_path: np.ndarray, xs: np.ndarray | None = None) -> np.ndarray:
    R = np.asarray(bloch_path, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 3:
        raise ValueError("bloch_path must have shape (n>=3, 3)")
    _, dx = _validated_grid(len(R), xs)
    return np.diff(R, axis=0) / dx[:, None]


def weighted_tangent_covariance(
    bloch_path: np.ndarray,
    weights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
) -> np.ndarray:
    """Quadrature-consistent covariance sum w_i D_i D_i^T dx_i / sum w_i dx_i."""
    R = np.asarray(bloch_path, dtype=float)
    D = bloch_tangents(R, xs)
    _, dx = _validated_grid(len(R), xs)
    if weights is None:
        w = np.ones(len(D), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != len(D):
            raise ValueError("weights must have one value per trajectory increment")
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
    quadrature = w * dx
    denom = float(np.sum(quadrature))
    if denom <= 0:
        return np.zeros((3, 3), dtype=float)
    return np.einsum("i,ij,ik->jk", quadrature, D, D) / denom


def weighted_tangent_energy(
    bloch_path: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
) -> float:
    C = weighted_tangent_covariance(bloch_path, weights=weights, xs=xs)
    axis = _canonical_axis(v)
    return float(axis @ C @ axis)


def quaternion_motion(quaternions: np.ndarray) -> np.ndarray:
    """Raw relative SO(3) rotation increments between consecutive quaternions."""
    Q = np.asarray(quaternions, dtype=float)
    if Q.ndim != 2 or Q.shape[1] != 4 or Q.shape[0] < 2:
        raise ValueError("quaternions must have shape (n>=2, 4)")
    return quaternion_path_distances(Q)


def state_motion(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=complex)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must have shape (n>=2, state_dimension)")
    return np.asarray([
        fubini_study_distance(states[i], states[i + 1])
        for i in range(len(states) - 1)
    ])


def readout_values(bloch_path: np.ndarray, v: np.ndarray) -> np.ndarray:
    R = np.asarray(bloch_path, dtype=float)
    return R @ _canonical_axis(v)


def readout_variations(bloch_path: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(readout_values(bloch_path, v)))


def select_axis(
    bloch_path: np.ndarray,
    *,
    weights: np.ndarray | None,
    xs: np.ndarray | None,
    method: str,
    energy_tol: float = 1e-12,
    eigengap_tol: float = 1e-6,
) -> AxisSelectionResult:
    C = weighted_tangent_covariance(bloch_path, weights=weights, xs=xs)
    return principal_axis_result(
        C,
        method=method,
        energy_tol=energy_tol,
        eigengap_tol=eigengap_tol,
    )


def unweighted_bloch_pca_axis(
    bloch_path: np.ndarray,
    xs: np.ndarray | None = None,
    **thresholds: float,
) -> AxisSelectionResult:
    return select_axis(
        bloch_path,
        weights=None,
        xs=xs,
        method="unweighted_bloch_pca",
        **thresholds,
    )


def state_motion_weighted_axis(
    bloch_path: np.ndarray,
    states: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
    **thresholds: float,
) -> AxisSelectionResult:
    return select_axis(
        bloch_path,
        weights=state_motion(states) ** power,
        xs=xs,
        method="state_motion_weighted",
        **thresholds,
    )


def raw_quaternion_weighted_axis(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
    **thresholds: float,
) -> AxisSelectionResult:
    return select_axis(
        bloch_path,
        weights=quaternion_motion(quaternions) ** power,
        xs=xs,
        method="raw_quaternion_weighted",
        **thresholds,
    )


def quotient_quaternion_weighted_axis(
    bloch_path: np.ndarray,
    states: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
    **thresholds: float,
) -> AxisSelectionResult:
    """Gauge-aware axis; quotient motion equals Fubini--Study state motion."""
    return select_axis(
        bloch_path,
        weights=state_motion(states) ** power,
        xs=xs,
        method="quotient_quaternion_weighted",
        **thresholds,
    )


def best_pauli_axis(
    bloch_path: np.ndarray,
    weights: np.ndarray | None = None,
    xs: np.ndarray | None = None,
) -> AxisSelectionResult:
    axes = np.eye(3)
    energies = np.asarray([
        weighted_tangent_energy(bloch_path, axis, weights=weights, xs=xs)
        for axis in axes
    ])
    index = int(np.argmax(energies))
    return AxisSelectionResult(
        axis=axes[index],
        eigenvalues=np.sort(energies)[::-1],
        eigengap=float((np.max(energies) - np.partition(energies, -2)[-2]) / (np.max(energies) + 1e-12)),
        identifiable=bool(np.max(energies) > 1e-12),
        energy=float(energies[index]),
        method="best_pauli",
    )


# Backward-compatible raw-quaternion API used by legacy experiments.
def quaternion_weighted_covariance(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> np.ndarray:
    return weighted_tangent_covariance(
        bloch_path,
        weights=quaternion_motion(quaternions) ** power,
        xs=xs,
    )


def quaternion_guided_axis(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> np.ndarray:
    result = raw_quaternion_weighted_axis(
        bloch_path,
        quaternions,
        xs=xs,
        power=power,
        eigengap_tol=-1.0,
    )
    if result.axis is None:
        raise ValueError("trajectory does not define an identifiable axis")
    return result.axis


def quaternion_weighted_energy(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    xs: np.ndarray | None = None,
    power: float = 2.0,
) -> float:
    return weighted_tangent_energy(
        bloch_path,
        v,
        weights=quaternion_motion(quaternions) ** power,
        xs=xs,
    )


def hidden_motion_ratio(
    bloch_path: np.ndarray,
    quaternions: np.ndarray,
    v: np.ndarray,
    eta: float = 1e-12,
) -> float:
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
    dq = quaternion_motion(quaternions)
    dv = readout_variations(bloch_path, v)
    return float(np.sum(dv**2) / (np.sum(dq**2) + eta))


def observable_hidden_motion_ratio(
    bloch_path: np.ndarray,
    states: np.ndarray,
    v: np.ndarray,
    eta: float = 1e-12,
) -> float:
    """Gauge-invariant state-motion energy divided by visible readout variation."""
    ds = state_motion(states)
    dv = readout_variations(bloch_path, v)
    return float(np.sum(ds**2) / (np.sum(dv**2) + eta))


def axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between axes, identifying v and -v."""
    aa = _canonical_axis(a)
    bb = _canonical_axis(b)
    return float(np.arccos(np.clip(abs(float(aa @ bb)), -1.0, 1.0)))
