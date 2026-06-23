from __future__ import annotations

import numpy as np

from qru_registerization.coherent_register import qae_signed_error_budget_from_probability


def test_downstream_operator_bound_on_dense_grid():
    worst_slack = float("inf")
    worst_violation = 0.0
    for probability in np.linspace(0.0, 1.0, 201):
        for phase_bits, magnitude_bits in ((3, 2), (4, 3), (5, 4)):
            for gamma in (0.1, 0.5, 0.73, 1.0, 1.5):
                budget = qae_signed_error_budget_from_probability(
                    float(probability), phase_bits, magnitude_bits, gamma
                )
                slack = (
                    budget["downstream_lipschitz_bound"]
                    - budget["expected_downstream_operator_norm_error"]
                )
                worst_slack = min(worst_slack, slack)
                worst_violation = max(worst_violation, -slack)
    assert worst_violation <= 1e-12
    assert worst_slack >= -1e-12
