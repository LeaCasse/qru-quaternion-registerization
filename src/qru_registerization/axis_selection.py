from __future__ import annotations

import numpy as np


def _principal_eigenvector(C: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, int(np.argmax(vals))]
    # Deterministic sign convention: largest-magnitude component is positive.
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v / np.linalg.norm(v)


def variance_axis(bloch_path: np.ndarray) -> np.ndarray:
    """Axis maximizing variance of v.r over a Bloch trajectory."""
    R = np.asarray(bloch_path, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 2:
        raise ValueError("bloch_path must have shape (n>=2, 3)")
    centered = R - R.mean(axis=0, keepdims=True)
    C = centered.T @ centered / R.shape[0]
    return _principal_eigenvector(C)


def tangent_axis(bloch_path: np.ndarray, xs: np.ndarray | None = None) -> np.ndarray:
    """Axis maximizing projected tangent energy along a Bloch trajectory."""
    R = np.asarray(bloch_path, dtype=float)
    if R.ndim != 2 or R.shape[1] != 3 or R.shape[0] < 3:
        raise ValueError("bloch_path must have shape (n>=3, 3)")
    if xs is None:
        D = np.diff(R, axis=0)
    else:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        if len(xs) != len(R):
            raise ValueError("xs and bloch_path must have same length")
        dx = np.diff(xs)
        if np.any(dx == 0):
            raise ValueError("xs must be strictly varying")
        D = np.diff(R, axis=0) / dx[:, None]
    C = D.T @ D / D.shape[0]
    return _principal_eigenvector(C)


def projected_variance(bloch_path: np.ndarray, v: np.ndarray) -> float:
    R = np.asarray(bloch_path, dtype=float)
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    return float(np.var(R @ v))


def projected_tangent_energy(bloch_path: np.ndarray, v: np.ndarray, xs: np.ndarray | None = None) -> float:
    R = np.asarray(bloch_path, dtype=float)
    v = np.asarray(v, dtype=float).reshape(3)
    v = v / np.linalg.norm(v)
    if xs is None:
        D = np.diff(R, axis=0)
    else:
        dx = np.diff(np.asarray(xs, dtype=float))
        D = np.diff(R, axis=0) / dx[:, None]
    return float(np.mean((D @ v) ** 2))
