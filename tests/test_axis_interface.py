from __future__ import annotations

import numpy as np
import pytest

from qru_registerization.pipeline import registerize_path
from qru_registerization.quaternion_diagnostics import (
    canonical_orient_axis,
    principal_axis_result,
    readout_values,
    unweighted_bloch_pca_axis,
)


def test_canonical_orient_axis_sign_invariant():
    vectors = [
        np.array([-0.2, 0.9, -0.3]),
        np.array([0.0, -0.4, 0.4]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    for v in vectors:
        a = canonical_orient_axis(v)
        b = canonical_orient_axis(-v)
        assert np.allclose(a, b, atol=1e-15)
        k = int(np.argmax(np.abs(a)))
        assert a[k] >= 0.0
        assert np.isclose(np.linalg.norm(a), 1.0)


def test_canonical_orient_axis_rejects_invalid_axes():
    with pytest.raises(ValueError):
        canonical_orient_axis(np.zeros(3))
    with pytest.raises(ValueError):
        canonical_orient_axis(np.array([1.0, np.nan, 0.0]))


def test_readout_values_use_canonical_orientation():
    R = np.eye(3)
    v = np.array([-2.0, 0.1, 0.2])
    assert np.allclose(readout_values(R, v), readout_values(R, -v))


def test_registerize_path_orients_axis_before_signed_register():
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    bloch = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    a = registerize_path(states, bloch, np.array([0.0, 0.0, -1.0]), m=3)
    b = registerize_path(states, bloch, np.array([0.0, 0.0, 1.0]), m=3)
    assert np.allclose(a["axis"], b["axis"])
    assert np.allclose(a["s"], b["s"])
    assert np.allclose(a["s_quant"], b["s_quant"])


def test_degenerate_principal_axis_is_rejected():
    C = np.diag([1.0, 1.0, 0.0])
    result = principal_axis_result(C, method="degenerate", eigengap_tol=1e-4)
    assert not result.identifiable
    assert result.axis is None


def test_constant_trajectory_is_rejected_by_axis_selector():
    xs = np.linspace(0.0, 1.0, 17)
    R = np.tile(np.array([0.0, 0.0, 1.0]), (len(xs), 1))
    result = unweighted_bloch_pca_axis(R, xs=xs)
    assert not result.identifiable
    assert result.axis is None
