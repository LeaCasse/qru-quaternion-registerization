from __future__ import annotations

import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis, quaternion_weighted_energy, hidden_motion_ratio


def test_vquat_maximizes_energy_against_fixed_axes():
    xs = np.linspace(-np.pi, np.pi, 101)
    params = random_qru_params(depth=5, seed=13, scale=0.9)
    paths = compute_paths(params, xs)
    R, Q = paths["bloch_direct"], paths["quaternions"]
    fixed = [np.eye(3)[i] for i in range(3)]
    vq = quaternion_guided_axis(R, Q, xs)
    Eq = quaternion_weighted_energy(R, Q, vq, xs)
    assert all(Eq + 1e-12 >= quaternion_weighted_energy(R, Q, v, xs) for v in fixed)


def test_hidden_motion_ratio_is_positive():
    xs = np.linspace(-np.pi, np.pi, 101)
    params = random_qru_params(depth=3, seed=5, scale=0.8)
    paths = compute_paths(params, xs)
    R, Q = paths["bloch_direct"], paths["quaternions"]
    assert hidden_motion_ratio(R, Q, np.array([0.0, 0.0, 1.0])) > 0

from qru_registerization.bloch import bloch_vector
from qru_registerization.gates import ry, rz
from qru_registerization.quaternion_diagnostics import (
    quotient_quaternion_weighted_axis,
    raw_quaternion_weighted_axis,
    state_motion_weighted_axis,
    unweighted_bloch_pca_axis,
    weighted_tangent_covariance,
)
from qru_registerization.quaternion_geometry import su2_to_quaternion


def test_covariance_is_symmetric_positive_semidefinite():
    xs = np.linspace(-1.0, 1.0, 71)
    params = random_qru_params(depth=4, seed=19, scale=0.7)
    paths = compute_paths(params, xs)
    C = weighted_tangent_covariance(paths["bloch_direct"], xs=xs)
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(C)) >= -1e-12


def test_constant_trajectory_is_not_identifiable():
    xs = np.linspace(0.0, 1.0, 31)
    R = np.tile(np.array([0.0, 0.0, 1.0]), (len(xs), 1))
    result = unweighted_bloch_pca_axis(R, xs=xs)
    assert not result.identifiable
    assert result.axis is None


def test_isotropic_xy_covariance_is_not_identifiable():
    xs = np.linspace(0.0, 2.0 * np.pi, 801)
    R = np.column_stack([np.cos(xs), np.sin(xs), np.zeros_like(xs)])
    result = unweighted_bloch_pca_axis(R, xs=xs, eigengap_tol=1e-3)
    assert not result.identifiable
    assert result.axis is None


def test_projection_blindness_axes_lie_in_xy_plane():
    xs = np.linspace(-np.pi / 3.0, np.pi / 3.0, 161)
    theta = np.arccos(0.35)
    ket0 = np.array([1.0, 0.0], dtype=complex)
    U = np.asarray([rz(float(x)) @ ry(float(theta)) for x in xs])
    states = np.asarray([u @ ket0 for u in U])
    R = np.asarray([bloch_vector(psi) for psi in states])
    Q = np.asarray([su2_to_quaternion(u) for u in U])
    results = [
        unweighted_bloch_pca_axis(R, xs=xs),
        state_motion_weighted_axis(R, states, xs=xs),
        raw_quaternion_weighted_axis(R, Q, xs=xs),
        quotient_quaternion_weighted_axis(R, states, xs=xs),
    ]
    assert np.ptp(R[:, 2]) < 1e-12
    assert all(r.identifiable and r.axis is not None for r in results)
    assert all(abs(float(r.axis[2])) < 1e-10 for r in results)


def test_nonuniform_grid_and_refinement_are_stable():
    coarse = np.linspace(-1.0, 1.0, 101) ** 3
    fine = np.linspace(-1.0, 1.0, 401) ** 3
    coarse.sort(); fine.sort()
    def path(xs):
        return np.column_stack([
            np.sqrt(1.0 - 0.2**2) * np.cos(xs),
            np.sqrt(1.0 - 0.2**2) * np.sin(xs),
            np.full_like(xs, 0.2),
        ])
    a = unweighted_bloch_pca_axis(path(coarse), xs=coarse)
    b = unweighted_bloch_pca_axis(path(fine), xs=fine)
    assert a.identifiable and b.identifiable
    assert abs(float(np.dot(a.axis, b.axis))) > 0.999


def test_quotient_and_state_weighted_axes_are_identical():
    from qru_registerization.gates import random_qru_params
    from qru_registerization.pipeline import compute_paths
    from qru_registerization.quaternion_diagnostics import (
        axis_angle,
        quotient_quaternion_weighted_axis,
        state_motion_weighted_axis,
    )

    xs = np.linspace(-np.pi, np.pi, 101)
    paths = compute_paths(random_qru_params(depth=4, seed=17, scale=0.9), xs)
    q = quotient_quaternion_weighted_axis(paths["bloch_direct"], paths["states"], xs=xs)
    s = state_motion_weighted_axis(paths["bloch_direct"], paths["states"], xs=xs)
    assert q.identifiable == s.identifiable
    assert q.axis is not None and s.axis is not None
    assert axis_angle(q.axis, s.axis) < 1e-12
