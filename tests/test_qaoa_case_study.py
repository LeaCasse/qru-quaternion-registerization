from __future__ import annotations

from qru_registerization.qaoa_toy import evaluate_qaoa_with_estimated_field


def test_qaoa_metrics_are_finite():
    out = evaluate_qaoa_with_estimated_field(h_true=0.3, h_est=0.25, grid_size=9)
    assert out["energy_gap_to_ground"] >= -1e-10
    assert 0.0 <= out["ground_state_probability"] <= 1.0
