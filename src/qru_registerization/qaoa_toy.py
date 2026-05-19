from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


Z0 = kron2(Z, I2)
Z1 = kron2(I2, Z)
X0 = kron2(X, I2)
X1 = kron2(I2, X)
ZZ = kron2(Z, Z)
PLUS2 = np.ones(4, dtype=complex) / 2.0


def ising_hamiltonian(h: float, fixed_field: float = 0.35, coupling: float = 1.0) -> np.ndarray:
    """Two-qubit Ising Hamiltonian used as a downstream QAOA toy layer.

    H(h) = -coupling Z0 Z1 - h Z0 - fixed_field Z1.
    The QRU readout h acts as a local field. Lower energy is better.
    """
    return -coupling * ZZ - float(h) * Z0 - float(fixed_field) * Z1


def mixer_hamiltonian() -> np.ndarray:
    return X0 + X1


def _unitary_from_diagonal_hamiltonian(H: np.ndarray, angle: float) -> np.ndarray:
    diag = np.diag(H).real
    return np.diag(np.exp(-1j * angle * diag))


def _mixer_unitary(beta: float) -> np.ndarray:
    # exp(-i beta (X0+X1)) = RX-like on each qubit; direct eigensafe formula.
    vals, vecs = np.linalg.eigh(mixer_hamiltonian())
    return vecs @ np.diag(np.exp(-1j * beta * vals)) @ vecs.conj().T


def qaoa_p1_state(h_for_angles: float, gamma: float, beta: float) -> np.ndarray:
    Uc = _unitary_from_diagonal_hamiltonian(ising_hamiltonian(h_for_angles), gamma)
    Um = _mixer_unitary(beta)
    return Um @ Uc @ PLUS2


def expected_energy(state: np.ndarray, h_true: float) -> float:
    H = ising_hamiltonian(h_true)
    return float(np.vdot(state, H @ state).real)


def ground_state_probability(state: np.ndarray, h_true: float) -> float:
    H = ising_hamiltonian(h_true)
    diag = np.diag(H).real
    emin = np.min(diag)
    mask = np.isclose(diag, emin)
    probs = np.abs(state) ** 2
    return float(np.sum(probs[mask]))


def optimize_qaoa_p1_grid(h_est: float, grid_size: int = 41) -> dict[str, float]:
    """Grid-search p=1 QAOA angles for the estimated Hamiltonian H(h_est)."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    gammas = np.linspace(0.0, np.pi, grid_size)
    betas = np.linspace(0.0, np.pi / 2.0, grid_size)
    best = {"energy_est": np.inf, "gamma": 0.0, "beta": 0.0}
    for gamma in gammas:
        for beta in betas:
            psi = qaoa_p1_state(h_est, float(gamma), float(beta))
            e = expected_energy(psi, h_est)
            if e < best["energy_est"]:
                best = {"energy_est": float(e), "gamma": float(gamma), "beta": float(beta)}
    return best


def ground_energy(h_true: float) -> float:
    """Exact ground energy of the diagonal downstream Ising Hamiltonian."""
    return float(np.min(np.diag(ising_hamiltonian(h_true)).real))


def evaluate_qaoa_with_estimated_field(
    h_true: float,
    h_est: float,
    grid_size: int = 41,
    true_optimum: dict[str, float] | None = None,
) -> dict[str, float]:
    """Optimize QAOA using h_est, then evaluate under true H(h_true).

    The primary downstream metric is energy_gap_to_ground, which is always
    non-negative. We also report p1_reference_gap for information only; it can
    be negative because a circuit built with the wrong estimated Hamiltonian is
    not restricted to the same p=1 ansatz family as the reference circuit built
    with h_true.
    """
    opt_est = optimize_qaoa_p1_grid(h_est, grid_size=grid_size)
    psi_est = qaoa_p1_state(h_est, opt_est["gamma"], opt_est["beta"])
    energy_true = expected_energy(psi_est, h_true)
    gs_prob = ground_state_probability(psi_est, h_true)
    e_ground = ground_energy(h_true)

    if true_optimum is None:
        true_optimum = optimize_qaoa_p1_grid(h_true, grid_size=grid_size)
    psi_true = qaoa_p1_state(h_true, true_optimum["gamma"], true_optimum["beta"])
    energy_opt_true = expected_energy(psi_true, h_true)
    return {
        "h_true": float(h_true),
        "h_est": float(h_est),
        "gamma_est": opt_est["gamma"],
        "beta_est": opt_est["beta"],
        "energy_true": float(energy_true),
        "ground_energy": float(e_ground),
        "energy_gap_to_ground": float(energy_true - e_ground),
        "energy_opt_true": float(energy_opt_true),
        "p1_reference_gap": float(energy_true - energy_opt_true),
        "ground_state_probability": float(gs_prob),
    }
