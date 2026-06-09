from __future__ import annotations

import numpy as np

from qru_registerization.gates import random_qru_params, ry, rz
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_geometry import (
    bloch_geodesic_distance,
    fubini_study_distance,
    quotient_quaternion_distance,
    relative_rotation_angle,
    su2_to_quaternion,
)


def test_quaternion_bloch_consistency():
    xs = np.linspace(-1.0, 1.0, 25)
    params = random_qru_params(depth=4, seed=3, scale=0.7)
    paths = compute_paths(params, xs)
    assert np.allclose(np.linalg.norm(paths["quaternions"], axis=1), 1.0, atol=1e-10)
    assert np.allclose(paths["bloch_direct"], paths["bloch_quaternion"], atol=1e-10)


def test_quaternion_sign_invariance():
    q1 = su2_to_quaternion(ry(0.3) @ rz(-0.4))
    q2 = su2_to_quaternion(ry(-0.7) @ rz(0.2))
    assert np.isclose(relative_rotation_angle(q1, q2), relative_rotation_angle(q1, -q2))


def test_gauge_only_unitary_motion():
    V = ry(0.7)
    U0 = V @ rz(-1.1)
    U1 = V @ rz(1.4)
    q0 = su2_to_quaternion(U0)
    q1 = su2_to_quaternion(U1)
    raw = relative_rotation_angle(q0, q1)
    quotient = quotient_quaternion_distance(U0, U1)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    state = fubini_study_distance(U0 @ ket0, U1 @ ket0)
    assert raw > 1.0
    assert quotient < 1e-10
    assert state < 1e-10


def test_quotient_distance_matches_state_and_bloch_distance():
    Ua = ry(0.31) @ rz(-0.6)
    Ub = rz(0.23) @ ry(-0.42)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    psi_a = Ua @ ket0
    psi_b = Ub @ ket0
    # Direct Bloch conversion avoids assumptions about a particular QRU parameterization.
    from qru_registerization.bloch import bloch_vector
    d_q = quotient_quaternion_distance(Ua, Ub)
    d_fs = fubini_study_distance(psi_a, psi_b)
    d_b = bloch_geodesic_distance(bloch_vector(psi_a), bloch_vector(psi_b))
    assert np.isclose(d_q, d_fs, atol=1e-10)
    assert np.isclose(d_q, d_b, atol=1e-10)


def test_relative_rotation_angle_range():
    rng = np.random.default_rng(8)
    for _ in range(50):
        q1 = rng.normal(size=4)
        q2 = rng.normal(size=4)
        d = relative_rotation_angle(q1, q2)
        assert 0.0 <= d <= np.pi
