from __future__ import annotations

import numpy as np

Array = np.ndarray


def rx(theta: float) -> Array:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> Array:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> Array:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=complex,
    )


def random_qru_params(depth: int, seed: int = 7, scale: float = 1.0) -> Array:
    """Return QRU parameters with shape (depth, 4).

    Per layer, parameters are [alpha, beta, gamma, delta] and the layer is
    RX(alpha) RY(beta*x + delta) RZ(gamma), applied left-to-right to the state.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=(depth, 4))


def qru_unitary(params: Array, x: float) -> Array:
    """Build a single-qubit QRU unitary.

    Layer convention applied to the state:
        |psi> <- RZ(gamma) RY(beta*x + delta) RX(alpha) |psi>

    Matrix products therefore accumulate as U_layer @ U.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (depth, 4)")
    U = np.eye(2, dtype=complex)
    for alpha, beta, gamma, delta in params:
        layer = rz(gamma) @ ry(beta * x + delta) @ rx(alpha)
        U = layer @ U
    return U


def qru_state(params: Array, x: float) -> Array:
    """Return U_theta(x)|0>."""
    zero = np.array([1.0, 0.0], dtype=complex)
    return qru_unitary(params, x) @ zero


def assert_unitary(U: Array, atol: float = 1e-10) -> None:
    if not np.allclose(U.conj().T @ U, np.eye(2), atol=atol):
        raise ValueError("matrix is not unitary within tolerance")
