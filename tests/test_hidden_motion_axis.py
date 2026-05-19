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
