import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.axis_selection import variance_axis, tangent_axis, projected_variance, projected_tangent_energy


def test_variance_axis_beats_coordinate_axes():
    xs = np.linspace(-np.pi, np.pi, 151)
    paths = compute_paths(random_qru_params(depth=4, seed=21), xs)
    r = paths["bloch_direct"]
    v = variance_axis(r)
    best = projected_variance(r, v)
    for axis in np.eye(3):
        assert best + 1e-12 >= projected_variance(r, axis)


def test_tangent_axis_beats_coordinate_axes():
    xs = np.linspace(-np.pi, np.pi, 151)
    paths = compute_paths(random_qru_params(depth=4, seed=22), xs)
    r = paths["bloch_direct"]
    v = tangent_axis(r, xs)
    best = projected_tangent_energy(r, v, xs)
    for axis in np.eye(3):
        assert best + 1e-12 >= projected_tangent_energy(r, axis, xs)
