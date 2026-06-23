from __future__ import annotations

import numpy as np
import pytest
pytest.importorskip("qiskit")

from qru_registerization.coherent_register import (
    build_qae_signed_downstream_circuit,
    qae_signed_downstream_validation,
)
from qru_registerization.gates import ry

# Statevector validation is intentionally sparse; the dense error bound is tested analytically.
VALIDATION_CASES = (
    (0.0, (3, 2)),
    (1.0, (3, 2)),
    (0.125, (3, 2)),
    (0.5, (3, 2)),
    (0.3059, (4, 3)),
    (0.73, (4, 3)),
    (0.97, (4, 3)),
    (0.3059, (5, 4)),
)


@pytest.mark.parametrize("probability,precision", VALIDATION_CASES)
def test_qae_signed_pipeline_uncomputes_on_probability_grid(probability: float, precision: tuple[int, int]):
    phase_bits, magnitude_bits = precision
    amplitude_unitary = ry(2.0 * np.arcsin(np.sqrt(probability)))
    metrics = qae_signed_downstream_validation(
        amplitude_unitary,
        phase_bits=phase_bits,
        magnitude_bits=magnitude_bits,
        gamma=0.73,
    )
    assert metrics["state_fidelity"] > 1.0 - 1e-10
    assert metrics["signed_zero_probability"] > 1.0 - 1e-10
    assert metrics["signed_register_purity"] > 1.0 - 1e-10


def test_qae_signed_pipeline_has_no_intermediate_measurement_or_reset():
    circuit = build_qae_signed_downstream_circuit(
        ry(2.0 * np.arcsin(np.sqrt(0.3059))),
        phase_bits=4,
        magnitude_bits=3,
        gamma=0.73,
    )
    names = [instruction.operation.name for instruction in circuit.data]
    assert "measure" not in names
    assert "reset" not in names
    assert not circuit.cregs
