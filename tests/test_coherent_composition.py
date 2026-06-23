from __future__ import annotations

import itertools

import numpy as np
import pytest
pytest.importorskip("qiskit")
from qiskit import QuantumCircuit

from qru_registerization.coherent_register import (
    apply_register_controlled_z_evolution,
    register_controlled_z_fidelity,
)



@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("gamma", [-0.73, 0.19, 1.17])
def test_register_controlled_rotation_all_bitstrings(m: int, gamma: float):
    fidelities = []
    for sign_bit in (0, 1):
        for bits in itertools.product((0, 1), repeat=m):
            fidelities.append(register_controlled_z_fidelity(sign_bit, bits, gamma))
    assert min(fidelities) > 1.0 - 1e-12


def test_register_controlled_rotation_rejects_overlapping_qubits():
    circuit = QuantumCircuit(3)
    with pytest.raises(ValueError):
        apply_register_controlled_z_evolution(
            circuit,
            sign_qubit=0,
            magnitude_qubits=[1],
            target_qubit=1,
            gamma=0.4,
        )


def test_register_controlled_rotation_rejects_nonfinite_gamma():
    circuit = QuantumCircuit(3)
    with pytest.raises(ValueError):
        apply_register_controlled_z_evolution(
            circuit,
            sign_qubit=0,
            magnitude_qubits=[1],
            target_qubit=2,
            gamma=np.nan,
        )


def test_two_branch_lookup_and_echo_preserve_coherence():
    from qru_registerization.coherent_register import two_branch_coherence_echo_metrics
    from qru_registerization.gates import random_qru_params

    params = random_qru_params(depth=3, seed=23, scale=0.8)
    metrics = two_branch_coherence_echo_metrics(
        params=params,
        x_zero=-1.1,
        x_one=0.9,
        axis=np.array([0.4, -0.7, 0.5916079783099616]),
        m=3,
        gamma=0.73,
    )
    assert metrics["codes"][0] != metrics["codes"][1]
    assert metrics["p_input_zero_coherent"] > 1.0 - 1e-12
    assert abs(metrics["p_input_zero_after_dephasing"] - 0.5) < 1e-12
    assert abs(metrics["interference_gap"] - 0.5) < 1e-12
    assert metrics["coherent_purity"] > 1.0 - 1e-12
    assert abs(metrics["dephased_purity"] - 0.5) < 1e-12


def test_two_branch_pipeline_contains_no_measurement():
    from qru_registerization.coherent_register import build_two_branch_pipeline_segments
    from qru_registerization.gates import random_qru_params

    params = random_qru_params(depth=2, seed=11, scale=0.6)
    preparation, echo_tail, _ = build_two_branch_pipeline_segments(
        params=params,
        x_zero=-0.4,
        x_one=0.8,
        axis=np.array([0.2, 0.9, -0.3872983346207417]),
        m=2,
        gamma=0.41,
    )
    operation_names = [instruction.operation.name for instruction in preparation.data]
    operation_names += [instruction.operation.name for instruction in echo_tail.data]
    assert "measure" not in operation_names
