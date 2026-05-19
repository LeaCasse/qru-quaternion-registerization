import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.axis_selection import variance_axis


def test_pipeline_outputs_are_consistent():
    xs = np.linspace(-1.0, 1.0, 31)
    paths = compute_paths(random_qru_params(depth=3, seed=4), xs)
    assert paths["quaternions"].shape == (31, 4)
    assert paths["bloch_direct"].shape == (31, 3)
    assert np.max(np.linalg.norm(paths["bloch_direct"] - paths["bloch_quaternion"], axis=1)) < 1e-10
    v = variance_axis(paths["bloch_direct"])
    reg = registerize_path(paths["states"], paths["bloch_direct"], v, m=5)
    assert np.max(np.abs(reg["p_from_s"] - reg["p_direct"])) < 1e-10
    assert np.max(np.abs(reg["s"] - reg["s_from_p"])) < 1e-10
