import numpy as np

from qru_registerization.gates import rx, ry, rz, random_qru_params, qru_unitary, qru_state
from qru_registerization.quaternion_geometry import su2_to_quaternion, bloch_from_quaternion
from qru_registerization.bloch import bloch_vector


def test_basic_rotations_have_unit_quaternions():
    for U in [rx(0.3), ry(-0.7), rz(1.2)]:
        q = su2_to_quaternion(U)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)


def test_quaternion_bloch_matches_state_bloch():
    params = random_qru_params(depth=5, seed=123, scale=0.9)
    for x in np.linspace(-np.pi, np.pi, 51):
        U = qru_unitary(params, x)
        q = su2_to_quaternion(U)
        r_quat = bloch_from_quaternion(q)
        r_state = bloch_vector(qru_state(params, x))
        assert np.allclose(r_quat, r_state, atol=1e-10)
