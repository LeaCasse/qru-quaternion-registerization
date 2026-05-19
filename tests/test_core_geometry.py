from __future__ import annotations

import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths


def test_quaternion_bloch_consistency():
    xs = np.linspace(-1.0, 1.0, 25)
    params = random_qru_params(depth=4, seed=3, scale=0.7)
    paths = compute_paths(params, xs)
    assert np.allclose(np.linalg.norm(paths["quaternions"], axis=1), 1.0, atol=1e-10)
    assert np.allclose(paths["bloch_direct"], paths["bloch_quaternion"], atol=1e-10)
