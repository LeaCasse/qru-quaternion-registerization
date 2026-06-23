from __future__ import annotations

import pytest
pytest.importorskip("qiskit")

import numpy as np

from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis
from qru_registerization.bloch import signed_coordinate
from qru_registerization.amplitude_interface import projector_probability, probability_from_signed_coordinate, signed_coordinate_from_probability
from qru_registerization.fixed_point import max_quantization_error_bound
from qru_registerization.coherent_register import (
    directional_amplitude_unitary,
    qae_phase_distribution,
    qae_phase_distribution_from_probability,
    qae_signed_error_budget,
    qae_signed_error_budget_from_probability,
)
from qru_registerization.gates import qru_unitary


def test_projector_identity_for_vquat():
    xs = np.linspace(-1.0, 1.0, 11)
    params = random_qru_params(depth=4, seed=4, scale=0.6)
    paths = compute_paths(params, xs)
    R, Q, states = paths["bloch_direct"], paths["quaternions"], paths["states"]
    vq = quaternion_guided_axis(R, Q, xs)
    for r, psi in zip(R, states):
        s = signed_coordinate(r, vq)
        p1 = probability_from_signed_coordinate(s)
        p2 = projector_probability(psi, vq)
        assert abs(p1 - p2) < 1e-10
        assert abs(s - signed_coordinate_from_probability(p2)) < 1e-10


def test_signed_fixed_point_all_bitstrings():
    from qru_registerization.fixed_point import (
        decode_signed_fixed_point,
        encode_signed_fixed_point,
    )

    for m in (1, 2, 3, 4):
        for sign in (0, 1):
            for value in range(2**m):
                bits = np.array([(value >> (m - 1 - k)) & 1 for k in range(m)])
                decoded = decode_signed_fixed_point(sign, bits)
                expected = value / (2**m)
                if sign and value:
                    expected *= -1
                assert decoded == expected
                encoded_sign, encoded_bits = encode_signed_fixed_point(decoded, m)
                assert np.array_equal(encoded_bits, bits)
                assert encoded_sign == (sign if value else 0)


def test_signed_fixed_point_boundaries_and_invalid_inputs():
    from qru_registerization.fixed_point import (
        decode_signed_fixed_point,
        encode_signed_fixed_point,
        max_quantization_error_bound,
    )

    for m in (2, 4, 8):
        max_mag = 1.0 - 2.0 ** (-m)
        assert decode_signed_fixed_point(*encode_signed_fixed_point(1.0, m)) == max_mag
        assert decode_signed_fixed_point(*encode_signed_fixed_point(-1.0, m)) == -max_mag
        assert decode_signed_fixed_point(*encode_signed_fixed_point(-0.0, m)) == 0.0
        grid = np.linspace(-1.0, 1.0, 1001)
        errors = [abs(decode_signed_fixed_point(*encode_signed_fixed_point(s, m)) - s) for s in grid]
        assert max(errors) <= max_quantization_error_bound(m) + 1e-12

    import pytest
    with pytest.raises(ValueError):
        decode_signed_fixed_point(2, np.array([0, 1]))
    with pytest.raises(ValueError):
        decode_signed_fixed_point(0, np.array([0, 2]))
    with pytest.raises(ValueError):
        encode_signed_fixed_point(np.nan, 3)


def test_basis_rotation_matches_directional_observable_and_projector():
    from qru_registerization.amplitude_interface import (
        basis_rotation_from_axis,
        basis_rotation_probability,
        directional_observable,
        projector_probability,
    )
    from qru_registerization.bloch import Z

    rng = np.random.default_rng(21)
    for _ in range(50):
        v = rng.normal(size=3)
        B = basis_rotation_from_axis(v)
        lhs = B.conj().T @ Z @ B
        rhs = directional_observable(v)
        assert np.allclose(lhs, rhs, atol=1e-12)
        assert np.allclose(B.conj().T @ B, np.eye(2), atol=1e-12)

        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi /= np.linalg.norm(psi)
        assert abs(basis_rotation_probability(psi, v) - projector_probability(psi, v)) < 1e-12


def test_minimal_coherent_qae_exact_control_amplitudes():
    from qru_registerization.coherent_register import qae_amplitude_summary
    from qru_registerization.gates import ry

    for probability in (0.0, 0.5, 1.0):
        theta = np.arcsin(np.sqrt(probability))
        amplitude_unitary = ry(2.0 * theta)
        summary = qae_amplitude_summary(amplitude_unitary, phase_bits=2)
        assert abs(summary["exact_probability"] - probability) < 1e-12
        assert abs(summary["mean_estimate"] - probability) < 1e-12
        supported = {
            index
            for index, weight in summary["distribution"].items()
            if weight > 1e-12
        }
        expected = {
            0.0: {0},
            0.5: {1, 3},
            1.0: {2},
        }[probability]
        assert supported == expected


def test_directional_amplitude_unitary_matches_projector_probability():
    from qru_registerization.amplitude_interface import projector_probability
    from qru_registerization.coherent_register import directional_amplitude_unitary
    from qru_registerization.gates import qru_state, qru_unitary, random_qru_params

    params = random_qru_params(depth=3, seed=17, scale=0.7)
    axis = np.array([0.3, -0.4, 0.8660254037844386])
    x = 0.42
    amplitude_unitary = directional_amplitude_unitary(qru_unitary(params, x), axis)
    qae_good_probability = abs(amplitude_unitary[1, 0]) ** 2
    expected = projector_probability(qru_state(params, x), axis)
    assert abs(qae_good_probability - expected) < 1e-12


def test_minimal_coherent_qae_contains_no_measurement():
    from qru_registerization.coherent_register import build_minimal_coherent_qae_circuit
    from qru_registerization.gates import ry

    circuit = build_minimal_coherent_qae_circuit(ry(np.pi / 3.0), phase_bits=2)
    assert "measure" not in [instruction.operation.name for instruction in circuit.data]


def test_phase_to_signed_lookup_all_phase_words():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qru_registerization.coherent_register import (
        apply_phase_to_signed_lookup,
        phase_index_to_signed_code,
    )

    phase_bits = 4
    magnitude_bits = 3
    phase_qubits = tuple(range(phase_bits))
    sign_qubit = phase_bits
    magnitude_qubits = tuple(range(phase_bits + 1, phase_bits + 1 + magnitude_bits))
    for index in range(2**phase_bits):
        circuit = QuantumCircuit(phase_bits + 1 + magnitude_bits)
        for j, qubit in enumerate(phase_qubits):
            if (index >> j) & 1:
                circuit.x(qubit)
        apply_phase_to_signed_lookup(circuit, phase_qubits, sign_qubit, magnitude_qubits)
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities_dict()
        assert len([value for value in probabilities.values() if value > 1e-12]) == 1
        bitstring = max(probabilities, key=probabilities.get)
        # Qiskit prints highest qubit first.
        actual_sign = int(bitstring[-1 - sign_qubit])
        actual_bits = np.array(
            [int(bitstring[-1 - qubit]) for qubit in magnitude_qubits], dtype=int
        )
        expected_sign, expected_bits, _, _ = phase_index_to_signed_code(
            index, phase_bits, magnitude_bits
        )
        assert actual_sign == expected_sign
        assert np.array_equal(actual_bits, expected_bits)

        # The decoder is self-inverse.
        apply_phase_to_signed_lookup(circuit, phase_qubits, sign_qubit, magnitude_qubits)
        final = Statevector.from_instruction(circuit)
        expected_index = index
        assert abs(final.probabilities()[expected_index] - 1.0) < 1e-12


def test_qae_signed_downstream_matches_phase_reference_and_uncomputes():
    from qru_registerization.coherent_register import qae_signed_downstream_validation
    from qru_registerization.gates import ry

    amplitude_unitary = ry(1.17)
    metrics = qae_signed_downstream_validation(
        amplitude_unitary, phase_bits=4, magnitude_bits=3, gamma=0.73
    )
    assert metrics["state_fidelity"] > 1.0 - 1e-12
    assert metrics["signed_zero_probability"] > 1.0 - 1e-12
    assert metrics["signed_register_purity"] > 1.0 - 1e-12


def test_qae_signed_downstream_contains_no_measurement():
    from qru_registerization.coherent_register import build_qae_signed_downstream_circuit
    from qru_registerization.gates import ry

    circuit = build_qae_signed_downstream_circuit(
        ry(1.17), phase_bits=4, magnitude_bits=3, gamma=0.73
    )
    assert "measure" not in [instruction.operation.name for instruction in circuit.data]


def test_qae_signed_error_budget_respects_downstream_bound():
    from qru_registerization.coherent_register import (
        directional_amplitude_unitary,
        qae_signed_error_budget,
    )
    from qru_registerization.gates import qru_unitary, random_qru_params

    params = random_qru_params(depth=2, seed=7, scale=0.5)
    A = directional_amplitude_unitary(
        qru_unitary(params, 0.37), np.array([0.2, -0.4, 0.8944271909999159])
    )
    result = qae_signed_error_budget(A, phase_bits=3, magnitude_bits=2, gamma=0.73)
    assert result["expected_downstream_operator_norm_error"] <= (
        result["downstream_lipschitz_bound"] + 1e-12
    )
    assert result["expected_total_signed_abs_error"] <= (
        result["expected_qae_signed_abs_error"]
        + result["expected_decoder_abs_error"]
        + 1e-12
    )


def test_analytic_qae_distribution_matches_statevector():
    rng = np.random.default_rng(77)
    for phase_bits in (2, 3, 4):
        for probability in rng.uniform(0.0, 1.0, size=8):
            angle = 2.0 * np.arcsin(np.sqrt(probability))
            amplitude_unitary = np.array([
                [np.cos(angle / 2.0), -np.sin(angle / 2.0)],
                [np.sin(angle / 2.0), np.cos(angle / 2.0)],
            ], dtype=complex)
            circuit_distribution = qae_phase_distribution(amplitude_unitary, phase_bits)
            analytic_distribution = qae_phase_distribution_from_probability(probability, phase_bits)
            assert max(abs(circuit_distribution[i] - analytic_distribution[i]) for i in circuit_distribution) < 1e-10


def test_analytic_error_budget_matches_circuit_budget():
    xs = np.linspace(-0.8, 0.8, 5)
    params = random_qru_params(depth=3, seed=19, scale=0.7)
    axis = np.array([0.2, 0.9, -0.3])
    axis /= np.linalg.norm(axis)
    for x in xs:
        amplitude_unitary = directional_amplitude_unitary(qru_unitary(params, float(x)), axis)
        probability = float(abs(amplitude_unitary[1, 0]) ** 2)
        for phase_bits, magnitude_bits in ((3, 2), (4, 3)):
            circuit = qae_signed_error_budget(amplitude_unitary, phase_bits, magnitude_bits, gamma=0.73)
            analytic = qae_signed_error_budget_from_probability(probability, phase_bits, magnitude_bits, gamma=0.73)
            for key in (
                "expected_qae_signed_abs_error",
                "expected_decoder_abs_error",
                "expected_total_signed_abs_error",
                "expected_downstream_operator_norm_error",
                "expected_downstream_state_infidelity",
            ):
                assert abs(circuit[key] - analytic[key]) < 1e-10
