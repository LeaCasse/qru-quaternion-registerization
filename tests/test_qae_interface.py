from __future__ import annotations

import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis
from qru_registerization.bloch import signed_coordinate
from qru_registerization.amplitude_interface import projector_probability, probability_from_signed_coordinate, signed_coordinate_from_probability
from qru_registerization.readout_estimators import ideal_qae_register_estimate
from qru_registerization.fixed_point import max_quantization_error_bound


def test_projector_identity_for_vquat():
    xs = np.linspace(-1.0, 1.0, 11)
    params = random_qru_params(depth=4, seed=4, scale=0.6)
    paths = compute_paths(params, xs)
    R, Q, states = paths["bloch_direct"], paths["quaternions"], paths["states"]
    vq = quaternion_guided_axis(R, Q, xs)
    for r, psi in zip(R, states):
        s = signed_coordinate(r, vq)
        p1 = probability_from_signed_coordinate(s)
        p2 = projector_probability(psi, vq)
        assert abs(p1 - p2) < 1e-10
        assert abs(s - signed_coordinate_from_probability(p2)) < 1e-10


def test_qae_register_error_bound():
    s = 0.472
    eps_p = 1 / 128
    m = 8
    est = ideal_qae_register_estimate(s, epsilon_p=eps_p, m_bits=m, seed=1)
    assert abs(est["s_quant"] - s) <= 2 * eps_p + max_quantization_error_bound(m) + 1e-12
