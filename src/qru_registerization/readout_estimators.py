from __future__ import annotations

import numpy as np

from .amplitude_interface import probability_from_signed_coordinate, signed_coordinate_from_probability, ideal_ae_error_model
from .fixed_point import quantize_signed_magnitude


def shot_estimate_signed_coordinate(s: float, shots: int, seed: int | None = None) -> float:
    """Estimate signed coordinate s in [-1,1] by projective ±1 sampling.

    Outcomes are +1 with probability p=(1+s)/2 and -1 otherwise.
    The returned value is the empirical mean.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    p = probability_from_signed_coordinate(float(s))
    rng = np.random.default_rng(seed)
    plus = rng.binomial(shots, p)
    return float((2 * plus - shots) / shots)


def ideal_qae_register_estimate(
    s: float,
    epsilon_p: float,
    m_bits: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Toy QAE-compatible signed register estimate.

    This is not a hardware QAE implementation. It models a coherent amplitude
    estimator returning p_tilde with |p_tilde-p| <= epsilon_p, followed by the
    affine map s=2p-1 and signed-magnitude fixed-point quantization.
    """
    if epsilon_p < 0:
        raise ValueError("epsilon_p must be non-negative")
    p = probability_from_signed_coordinate(float(s))
    p_tilde = ideal_ae_error_model(p, epsilon=epsilon_p, seed=seed)
    s_tilde = signed_coordinate_from_probability(p_tilde)
    s_quant = quantize_signed_magnitude(s_tilde, m_bits)
    return {
        "p_exact": float(p),
        "p_tilde": float(p_tilde),
        "s_tilde": float(s_tilde),
        "s_quant": float(s_quant),
    }


def proxy_sampling_cost_for_precision(epsilon: float) -> float:
    """Shot sampling proxy O(1/epsilon^2)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(1.0 / epsilon**2)


def proxy_qae_query_cost_for_precision(epsilon: float) -> float:
    """Amplitude-estimation query proxy O(1/epsilon)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(1.0 / epsilon)
