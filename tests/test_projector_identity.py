import numpy as np

from qru_registerization.gates import random_qru_params, qru_state
from qru_registerization.bloch import signed_coordinate
from qru_registerization.amplitude_interface import projector_probability, probability_from_signed_coordinate


def test_projector_identity_random_states_axes():
    rng = np.random.default_rng(42)
    params = random_qru_params(depth=4, seed=9, scale=1.0)
    for x in np.linspace(-2.0, 2.0, 25):
        psi = qru_state(params, x)
        for _ in range(10):
            v = rng.normal(size=3)
            s = signed_coordinate(psi, v)
            p1 = probability_from_signed_coordinate(s)
            p2 = projector_probability(psi, v)
            assert np.isclose(p1, p2, atol=1e-10)
