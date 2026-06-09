from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from .fixed_point import decode_signed_fixed_point


def _validate_qubit_layout(
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
    target_qubit: int,
) -> tuple[int, ...]:
    magnitude_qubits = tuple(int(q) for q in magnitude_qubits)
    if not magnitude_qubits:
        raise ValueError("magnitude_qubits must be non-empty")
    all_qubits = (int(sign_qubit), *magnitude_qubits, int(target_qubit))
    if len(set(all_qubits)) != len(all_qubits):
        raise ValueError("sign, magnitude, and target qubits must be distinct")
    if min(all_qubits) < 0:
        raise ValueError("qubit indices must be non-negative")
    return magnitude_qubits


def prepare_signed_fixed_point_basis(
    circuit: QuantumCircuit,
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
    sign_bit: int,
    bits: Sequence[int],
) -> None:
    """Prepare a signed-magnitude computational-basis register."""
    magnitude_qubits = tuple(int(q) for q in magnitude_qubits)
    bits_array = np.asarray(bits, dtype=int)
    if sign_bit not in (0, 1):
        raise ValueError("sign_bit must be 0 or 1")
    if bits_array.ndim != 1 or bits_array.size != len(magnitude_qubits):
        raise ValueError("bits must match magnitude_qubits")
    if not np.all(np.isin(bits_array, (0, 1))):
        raise ValueError("bits must be binary")

    if sign_bit:
        circuit.x(sign_qubit)
    for qubit, bit in zip(magnitude_qubits, bits_array, strict=True):
        if bit:
            circuit.x(qubit)


def apply_register_controlled_z_evolution(
    circuit: QuantumCircuit,
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
    target_qubit: int,
    gamma: float,
) -> None:
    r"""Apply ``exp(-i gamma s Z)`` controlled by a signed-magnitude register.

    The magnitude qubits are ordered from the ``2^-1`` bit to the ``2^-m`` bit.
    Qiskit's ``RZ(theta)`` equals ``exp(-i theta Z / 2)``, hence each magnitude
    bit controls an ``RZ(2 gamma 2^-k)`` rotation. The sign qubit selects the
    rotation direction without classical conversion of the register value.
    """
    magnitude_qubits = _validate_qubit_layout(
        sign_qubit, magnitude_qubits, target_qubit
    )
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")

    for k, magnitude_qubit in enumerate(magnitude_qubits, start=1):
        angle = 2.0 * float(gamma) * (2.0 ** (-k))

        # Positive branch: controls are |sign=0, magnitude_k=1>.
        circuit.x(sign_qubit)
        circuit.mcrz(angle, [sign_qubit, magnitude_qubit], target_qubit)
        circuit.x(sign_qubit)

        # Negative branch: controls are |sign=1, magnitude_k=1>.
        circuit.mcrz(-angle, [sign_qubit, magnitude_qubit], target_qubit)


def build_register_controlled_z_circuit(
    sign_bit: int,
    bits: Sequence[int],
    gamma: float,
) -> QuantumCircuit:
    """Build the minimal basis-register downstream circuit used in validation."""
    bits_array = np.asarray(bits, dtype=int)
    if bits_array.ndim != 1 or bits_array.size == 0:
        raise ValueError("bits must be a non-empty one-dimensional sequence")
    m = int(bits_array.size)
    sign_qubit = 0
    magnitude_qubits = tuple(range(1, m + 1))
    target_qubit = m + 1
    circuit = QuantumCircuit(m + 2, name="signed_register_z_evolution")
    prepare_signed_fixed_point_basis(
        circuit,
        sign_qubit,
        magnitude_qubits,
        sign_bit,
        bits_array,
    )
    circuit.h(target_qubit)
    apply_register_controlled_z_evolution(
        circuit,
        sign_qubit,
        magnitude_qubits,
        target_qubit,
        gamma,
    )
    return circuit


def build_direct_z_reference_circuit(
    sign_bit: int,
    bits: Sequence[int],
    gamma: float,
) -> QuantumCircuit:
    """Build the exact reference using a direct rotation by the decoded value."""
    bits_array = np.asarray(bits, dtype=int)
    m = int(bits_array.size)
    sign_qubit = 0
    magnitude_qubits = tuple(range(1, m + 1))
    target_qubit = m + 1
    circuit = QuantumCircuit(m + 2, name="direct_z_reference")
    prepare_signed_fixed_point_basis(
        circuit,
        sign_qubit,
        magnitude_qubits,
        sign_bit,
        bits_array,
    )
    circuit.h(target_qubit)
    signed_value = decode_signed_fixed_point(sign_bit, bits_array)
    circuit.rz(2.0 * float(gamma) * signed_value, target_qubit)
    return circuit


def register_controlled_z_fidelity(
    sign_bit: int,
    bits: Sequence[int],
    gamma: float,
) -> float:
    """Return statevector fidelity against the direct ``exp(-i gamma s Z)`` reference."""
    controlled = Statevector.from_instruction(
        build_register_controlled_z_circuit(sign_bit, bits, gamma)
    )
    reference = Statevector.from_instruction(
        build_direct_z_reference_circuit(sign_bit, bits, gamma)
    )
    return float(state_fidelity(reference, controlled))


def _apply_controlled_unitary_branch(
    circuit: QuantumCircuit,
    control_qubit: int,
    target_qubit: int,
    unitary: np.ndarray,
    control_value: int,
) -> None:
    """Apply a one-qubit unitary on one branch of a binary control."""
    from qiskit.circuit.library import UnitaryGate

    if control_value not in (0, 1):
        raise ValueError("control_value must be 0 or 1")
    gate = UnitaryGate(np.asarray(unitary, dtype=complex)).control(1)
    if control_value == 0:
        circuit.x(control_qubit)
    circuit.append(gate, [control_qubit, target_qubit])
    if control_value == 0:
        circuit.x(control_qubit)


def apply_two_branch_signed_lookup(
    circuit: QuantumCircuit,
    input_qubit: int,
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
    code_zero: tuple[int, Sequence[int]],
    code_one: tuple[int, Sequence[int]],
) -> None:
    """Reversibly write one of two signed-magnitude words controlled by input.

    The register must initially be in ``|0...0>``. The operation is its own
    inverse because it consists only of branch-controlled X gates.
    """
    magnitude_qubits = tuple(int(q) for q in magnitude_qubits)
    all_targets = (int(sign_qubit), *magnitude_qubits)
    if input_qubit in all_targets or len(set(all_targets)) != len(all_targets):
        raise ValueError("input and register qubits must be distinct")

    for control_value, (sign_bit, bits) in enumerate((code_zero, code_one)):
        bits_array = np.asarray(bits, dtype=int)
        if sign_bit not in (0, 1):
            raise ValueError("sign bits must be binary")
        if bits_array.shape != (len(magnitude_qubits),) or not np.all(
            np.isin(bits_array, (0, 1))
        ):
            raise ValueError("lookup magnitude words must match the register")
        word = (int(sign_bit), *tuple(int(b) for b in bits_array))
        for target, bit in zip(all_targets, word, strict=True):
            if not bit:
                continue
            if control_value == 0:
                circuit.x(input_qubit)
            circuit.cx(input_qubit, target)
            if control_value == 0:
                circuit.x(input_qubit)


def build_two_branch_pipeline_segments(
    params: np.ndarray,
    x_zero: float,
    x_one: float,
    axis: np.ndarray,
    m: int,
    gamma: float,
) -> tuple[QuantumCircuit, QuantumCircuit, dict[str, object]]:
    """Build preparation and echo-tail segments for a two-input coherent demo.

    The signed register is written by a reversible two-entry lookup compiled
    from the exact QRU readouts. This validates composability and coherence;
    it is not an amplitude-estimation implementation.
    """
    from .bloch import bloch_vector
    from .fixed_point import decode_signed_fixed_point, encode_signed_fixed_point
    from .gates import qru_state, qru_unitary

    if m <= 0:
        raise ValueError("m must be positive")
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be a finite three-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        raise ValueError("axis must be nonzero")
    axis = axis / norm

    input_qubit = 0
    qru_qubit = 1
    sign_qubit = 2
    magnitude_qubits = tuple(range(3, 3 + m))
    target_qubit = 3 + m
    n_qubits = 4 + m

    states = [qru_state(params, float(x)) for x in (x_zero, x_one)]
    signed_values = [float(axis @ bloch_vector(state)) for state in states]
    codes = [encode_signed_fixed_point(value, m) for value in signed_values]
    quantized_values = [decode_signed_fixed_point(sign, bits) for sign, bits in codes]

    branch = QuantumCircuit(n_qubits, name="qru_lookup")
    _apply_controlled_unitary_branch(
        branch, input_qubit, qru_qubit, qru_unitary(params, x_zero), 0
    )
    _apply_controlled_unitary_branch(
        branch, input_qubit, qru_qubit, qru_unitary(params, x_one), 1
    )
    apply_two_branch_signed_lookup(
        branch,
        input_qubit,
        sign_qubit,
        magnitude_qubits,
        codes[0],
        codes[1],
    )

    preparation = QuantumCircuit(n_qubits, name="coherent_preparation")
    preparation.h(input_qubit)
    preparation.compose(branch, inplace=True)
    preparation.h(target_qubit)

    downstream = QuantumCircuit(n_qubits, name="register_downstream")
    apply_register_controlled_z_evolution(
        downstream,
        sign_qubit,
        magnitude_qubits,
        target_qubit,
        gamma,
    )

    echo_tail = QuantumCircuit(n_qubits, name="coherent_echo_tail")
    echo_tail.compose(downstream, inplace=True)
    echo_tail.compose(downstream.inverse(), inplace=True)
    echo_tail.h(target_qubit)
    echo_tail.compose(branch.inverse(), inplace=True)
    echo_tail.h(input_qubit)

    metadata: dict[str, object] = {
        "input_qubit": input_qubit,
        "qru_qubit": qru_qubit,
        "sign_qubit": sign_qubit,
        "magnitude_qubits": magnitude_qubits,
        "target_qubit": target_qubit,
        "signed_values": tuple(signed_values),
        "quantized_values": tuple(float(value) for value in quantized_values),
        "codes": tuple(
            (int(sign), tuple(int(bit) for bit in bits))
            for sign, bits in codes
        ),
    }
    return preparation, echo_tail, metadata


def _input_zero_probability(density_matrix: np.ndarray, input_qubit: int) -> float:
    diagonal = np.real(np.diag(density_matrix))
    indices = np.arange(diagonal.size)
    mask = ((indices >> int(input_qubit)) & 1) == 0
    return float(diagonal[mask].sum())


def two_branch_coherence_echo_metrics(
    params: np.ndarray,
    x_zero: float,
    x_one: float,
    axis: np.ndarray,
    m: int,
    gamma: float,
) -> dict[str, object]:
    """Compare coherent propagation with input dephasing at the interface."""
    from qiskit.quantum_info import DensityMatrix, Operator

    preparation, echo_tail, metadata = build_two_branch_pipeline_segments(
        params, x_zero, x_one, axis, m, gamma
    )
    prepared = DensityMatrix.from_instruction(preparation)

    input_qubit = int(metadata["input_qubit"])
    z_circuit = QuantumCircuit(preparation.num_qubits)
    z_circuit.z(input_qubit)
    z_operator = Operator(z_circuit).data
    rho = prepared.data
    dephased_rho = 0.5 * (rho + z_operator @ rho @ z_operator.conj().T)

    coherent_final = prepared.evolve(echo_tail).data
    measured_final = DensityMatrix(dephased_rho).evolve(echo_tail).data

    p0_coherent = _input_zero_probability(coherent_final, input_qubit)
    p0_measured = _input_zero_probability(measured_final, input_qubit)
    return {
        **metadata,
        "p_input_zero_coherent": p0_coherent,
        "p_input_zero_after_dephasing": p0_measured,
        "interference_gap": p0_coherent - p0_measured,
        "coherent_purity": float(np.real(np.trace(coherent_final @ coherent_final))),
        "dephased_purity": float(np.real(np.trace(measured_final @ measured_final))),
    }


def directional_amplitude_unitary(
    qru_matrix: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    r"""Return a one-qubit amplitude-preparation unitary for ``p_v``.

    The QRU prepares ``U|0>``. The directional basis rotation ``B_v`` maps
    the ``+1`` eigenspace of ``v.sigma`` to ``|0>``. A final ``X`` maps that
    probability to the standard QAE good state ``|1>``. Therefore
    ``|<1| A |0>|^2 = p_v = (1+s_v)/2``.
    """
    from .amplitude_interface import basis_rotation_from_axis

    qru_matrix = np.asarray(qru_matrix, dtype=complex)
    if qru_matrix.shape != (2, 2):
        raise ValueError("qru_matrix must be 2x2")
    if not np.allclose(qru_matrix.conj().T @ qru_matrix, np.eye(2), atol=1e-10):
        raise ValueError("qru_matrix must be unitary")
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return x_gate @ basis_rotation_from_axis(axis) @ qru_matrix


def amplitude_grover_operator(amplitude_unitary: np.ndarray) -> np.ndarray:
    r"""Return the one-qubit Grover operator used by standard QAE.

    For ``A|0> = sqrt(1-a)|0> + sqrt(a)e^{i phi}|1>``, the operator
    ``Q = -A S_0 A^dagger S_good`` has eigenvalues ``exp(+-2 i theta)`` with
    ``sin^2(theta)=a``.
    """
    amplitude_unitary = np.asarray(amplitude_unitary, dtype=complex)
    if amplitude_unitary.shape != (2, 2):
        raise ValueError("amplitude_unitary must be 2x2")
    if not np.allclose(
        amplitude_unitary.conj().T @ amplitude_unitary,
        np.eye(2),
        atol=1e-10,
    ):
        raise ValueError("amplitude_unitary must be unitary")
    s_zero = np.diag([-1.0, 1.0]).astype(complex)
    s_good = np.diag([1.0, -1.0]).astype(complex)
    return -amplitude_unitary @ s_zero @ amplitude_unitary.conj().T @ s_good


def build_minimal_coherent_qae_circuit(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 2,
) -> QuantumCircuit:
    """Build standard QPE-based amplitude estimation without measurement.

    Qubits ``0..phase_bits-1`` form the phase register and the final qubit is
    the amplitude system. The output remains coherent; measurement is left to
    the caller.
    """
    from qiskit.circuit.library import QFTGate, UnitaryGate

    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    amplitude_unitary = np.asarray(amplitude_unitary, dtype=complex)
    grover = amplitude_grover_operator(amplitude_unitary)
    system_qubit = phase_bits
    circuit = QuantumCircuit(phase_bits + 1, name=f"coherent_qae_m{phase_bits}")
    circuit.append(UnitaryGate(amplitude_unitary, label="A_v"), [system_qubit])
    for phase_qubit in range(phase_bits):
        circuit.h(phase_qubit)
        powered = np.linalg.matrix_power(grover, 2**phase_qubit)
        circuit.append(
            UnitaryGate(powered, label=f"Q^{2**phase_qubit}").control(1),
            [phase_qubit, system_qubit],
        )
    circuit.append(
        QFTGate(phase_bits).inverse(),
        list(range(phase_bits)),
    )
    return circuit


def qae_phase_distribution(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 2,
) -> dict[int, float]:
    """Return the exact phase-register distribution of the coherent QAE circuit."""
    circuit = build_minimal_coherent_qae_circuit(amplitude_unitary, phase_bits)
    state = Statevector.from_instruction(circuit)
    raw = state.probabilities_dict(qargs=list(range(phase_bits)))
    distribution = {int(str(bits), 2): float(prob) for bits, prob in raw.items()}
    return {index: distribution.get(index, 0.0) for index in range(2**phase_bits)}



def qae_phase_distribution_from_probability(
    probability: float,
    phase_bits: int,
) -> dict[int, float]:
    """Return the ideal QPE-based QAE phase distribution analytically.

    For ``p = sin^2(theta)`` the Grover eigenphases are
    ``phi = theta/pi`` and ``1-phi``. The prepared amplitude state has equal
    weight on the two eigenvectors, so the output distribution is the average
    of the two finite-QPE Dirichlet-kernel distributions.
    """
    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("probability must lie in [0, 1]")
    size = 2**phase_bits
    theta = float(np.arcsin(np.sqrt(float(probability))))
    phase = theta / np.pi

    def qpe_probability(index: int, eigenphase: float) -> float:
        delta = eigenphase - index / size
        denominator = np.sin(np.pi * delta)
        if abs(denominator) <= 1e-14:
            return 1.0
        numerator = np.sin(np.pi * size * delta)
        return float((numerator / (size * denominator)) ** 2)

    values = {
        index: 0.5 * (
            qpe_probability(index, phase)
            + qpe_probability(index, (1.0 - phase) % 1.0)
        )
        for index in range(size)
    }
    normalization = sum(values.values())
    return {index: value / normalization for index, value in values.items()}


def qae_signed_error_budget_from_probability(
    probability: float,
    phase_bits: int,
    magnitude_bits: int,
    gamma: float,
) -> dict[str, float]:
    """Return the signed-register error budget without constructing a circuit."""
    if magnitude_bits <= 0:
        raise ValueError("magnitude_bits must be positive")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    distribution = qae_phase_distribution_from_probability(probability, phase_bits)
    exact_s = 2.0 * float(probability) - 1.0
    qae_abs = decoder_abs = total_abs = downstream_norm = downstream_infidelity = 0.0
    for index, weight in distribution.items():
        phase_p = qae_amplitude_from_phase_index(index, phase_bits)
        phase_s = 2.0 * phase_p - 1.0
        _, _, decoded_s, _ = phase_index_to_signed_code(index, phase_bits, magnitude_bits)
        qae_abs += weight * abs(phase_s - exact_s)
        decoder_abs += weight * abs(decoded_s - phase_s)
        delta = decoded_s - exact_s
        total_abs += weight * abs(delta)
        downstream_norm += weight * 2.0 * abs(np.sin(0.5 * float(gamma) * delta))
        downstream_infidelity += weight * (np.sin(float(gamma) * delta) ** 2)
    return {
        "exact_probability": float(probability),
        "exact_signed_value": exact_s,
        "expected_qae_signed_abs_error": float(qae_abs),
        "expected_decoder_abs_error": float(decoder_abs),
        "expected_total_signed_abs_error": float(total_abs),
        "expected_downstream_operator_norm_error": float(downstream_norm),
        "downstream_lipschitz_bound": float(abs(gamma) * total_abs),
        "expected_downstream_state_infidelity": float(downstream_infidelity),
    }

def qae_amplitude_from_phase_index(index: int, phase_bits: int) -> float:
    """Map a QPE phase bin to the standard QAE amplitude estimate."""
    if phase_bits <= 0:
        raise ValueError("phase_bits must be positive")
    if index < 0 or index >= 2**phase_bits:
        raise ValueError("phase index outside the register range")
    return float(np.sin(np.pi * index / (2**phase_bits)) ** 2)


def qae_amplitude_summary(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 2,
) -> dict[str, object]:
    """Return exact amplitude, coherent QAE distribution, and decoded estimates."""
    amplitude_unitary = np.asarray(amplitude_unitary, dtype=complex)
    exact_probability = float(abs(amplitude_unitary[1, 0]) ** 2)
    distribution = qae_phase_distribution(amplitude_unitary, phase_bits)
    estimates = {
        index: qae_amplitude_from_phase_index(index, phase_bits)
        for index in distribution
    }
    mean_estimate = float(
        sum(distribution[index] * estimates[index] for index in distribution)
    )
    mode_index = max(distribution, key=distribution.get)
    return {
        "exact_probability": exact_probability,
        "distribution": distribution,
        "amplitude_estimates": estimates,
        "mean_estimate": mean_estimate,
        "mode_index": int(mode_index),
        "mode_estimate": float(estimates[mode_index]),
        "absolute_mean_error": abs(mean_estimate - exact_probability),
    }


def phase_index_to_signed_code(
    index: int,
    phase_bits: int,
    magnitude_bits: int,
) -> tuple[int, np.ndarray, float, float]:
    """Map a QAE phase word to a canonical signed fixed-point approximation.

    Returns ``(sign_bit, magnitude_word, decoded_value, exact_signed_value)``
    for ``s_y = 2 sin^2(pi y / 2^phase_bits) - 1``.
    """
    from .fixed_point import encode_signed_fixed_point, decode_signed_fixed_point

    if phase_bits <= 0 or magnitude_bits <= 0:
        raise ValueError("phase_bits and magnitude_bits must be positive")
    if index < 0 or index >= 2**phase_bits:
        raise ValueError("phase index outside the register range")
    probability = qae_amplitude_from_phase_index(index, phase_bits)
    exact_signed = 2.0 * probability - 1.0
    sign_bit, bits = encode_signed_fixed_point(exact_signed, magnitude_bits)
    decoded = decode_signed_fixed_point(sign_bit, bits)
    return int(sign_bit), np.asarray(bits, dtype=int), float(decoded), float(exact_signed)


def _apply_open_controlled_x(
    circuit: QuantumCircuit,
    control_qubits: Sequence[int],
    control_bits_lsb_first: Sequence[int],
    target_qubit: int,
) -> None:
    """Apply X to target iff controls equal the supplied LSB-first word."""
    controls = tuple(int(q) for q in control_qubits)
    bits = tuple(int(b) for b in control_bits_lsb_first)
    if len(controls) == 0 or len(controls) != len(bits):
        raise ValueError("control word must match a non-empty control register")
    if target_qubit in controls or len(set((*controls, int(target_qubit)))) != len(controls) + 1:
        raise ValueError("control and target qubits must be distinct")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("control bits must be binary")
    open_controls = [qubit for qubit, bit in zip(controls, bits, strict=True) if bit == 0]
    for qubit in open_controls:
        circuit.x(qubit)
    from qiskit.circuit.library import MCXGate
    circuit.append(MCXGate(len(controls)), [*controls, int(target_qubit)])
    for qubit in reversed(open_controls):
        circuit.x(qubit)


def apply_phase_to_signed_lookup(
    circuit: QuantumCircuit,
    phase_qubits: Sequence[int],
    sign_qubit: int,
    magnitude_qubits: Sequence[int],
) -> None:
    """Reversibly decode every QAE phase word into signed magnitude.

    ``phase_qubits`` are ordered least-significant first, matching the powers
    ``Q^(2^j)`` in :func:`build_minimal_coherent_qae_circuit`. The operation is
    self-inverse and assumes the signed output register starts in ``|0...0>``.
    """
    phase_qubits = tuple(int(q) for q in phase_qubits)
    magnitude_qubits = tuple(int(q) for q in magnitude_qubits)
    if not phase_qubits or not magnitude_qubits:
        raise ValueError("phase and magnitude registers must be non-empty")
    all_qubits = (*phase_qubits, int(sign_qubit), *magnitude_qubits)
    if len(set(all_qubits)) != len(all_qubits) or min(all_qubits) < 0:
        raise ValueError("phase and signed-register qubits must be distinct and non-negative")

    for index in range(2 ** len(phase_qubits)):
        sign_bit, bits, _, _ = phase_index_to_signed_code(
            index, len(phase_qubits), len(magnitude_qubits)
        )
        index_bits = tuple((index >> j) & 1 for j in range(len(phase_qubits)))
        output_word = (sign_bit, *tuple(int(bit) for bit in bits))
        output_qubits = (int(sign_qubit), *magnitude_qubits)
        for target, bit in zip(output_qubits, output_word, strict=True):
            if bit:
                _apply_open_controlled_x(circuit, phase_qubits, index_bits, target)


def _apply_phase_conditioned_z_evolution(
    circuit: QuantumCircuit,
    phase_qubits: Sequence[int],
    target_qubit: int,
    gamma: float,
    magnitude_bits: int,
) -> None:
    """Reference block-diagonal evolution directly controlled by phase words."""
    from qiskit.circuit.library import RZGate

    phase_qubits = tuple(int(q) for q in phase_qubits)
    if target_qubit in phase_qubits:
        raise ValueError("phase and target qubits must be distinct")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    for index in range(2 ** len(phase_qubits)):
        _, _, signed_value, _ = phase_index_to_signed_code(
            index, len(phase_qubits), magnitude_bits
        )
        if abs(signed_value) <= 1e-15:
            continue
        bits = tuple((index >> j) & 1 for j in range(len(phase_qubits)))
        open_controls = [q for q, bit in zip(phase_qubits, bits, strict=True) if bit == 0]
        for q in open_controls:
            circuit.x(q)
        controlled = RZGate(2.0 * float(gamma) * signed_value).control(len(phase_qubits))
        circuit.append(controlled, [*phase_qubits, int(target_qubit)])
        for q in reversed(open_controls):
            circuit.x(q)


def build_qae_signed_downstream_circuit(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 4,
    magnitude_bits: int = 3,
    gamma: float = 0.7,
    uncompute_signed: bool = True,
) -> QuantumCircuit:
    """Build QAE -> reversible signed decoder -> register-controlled Z evolution."""
    if phase_bits <= 0 or magnitude_bits <= 0:
        raise ValueError("phase_bits and magnitude_bits must be positive")
    phase_qubits = tuple(range(phase_bits))
    system_qubit = phase_bits
    sign_qubit = phase_bits + 1
    magnitude_qubits = tuple(range(phase_bits + 2, phase_bits + 2 + magnitude_bits))
    target_qubit = phase_bits + 2 + magnitude_bits
    circuit = QuantumCircuit(target_qubit + 1, name="qae_signed_downstream")
    qae = build_minimal_coherent_qae_circuit(amplitude_unitary, phase_bits)
    circuit.compose(qae, qubits=[*phase_qubits, system_qubit], inplace=True)
    apply_phase_to_signed_lookup(circuit, phase_qubits, sign_qubit, magnitude_qubits)
    circuit.h(target_qubit)
    apply_register_controlled_z_evolution(
        circuit, sign_qubit, magnitude_qubits, target_qubit, gamma
    )
    if uncompute_signed:
        apply_phase_to_signed_lookup(circuit, phase_qubits, sign_qubit, magnitude_qubits)
    return circuit


def build_qae_phase_reference_downstream_circuit(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 4,
    magnitude_bits: int = 3,
    gamma: float = 0.7,
) -> QuantumCircuit:
    """Reference using direct phase-word-controlled rotations and zero ancillas."""
    phase_qubits = tuple(range(phase_bits))
    system_qubit = phase_bits
    sign_qubit = phase_bits + 1
    magnitude_qubits = tuple(range(phase_bits + 2, phase_bits + 2 + magnitude_bits))
    target_qubit = phase_bits + 2 + magnitude_bits
    circuit = QuantumCircuit(target_qubit + 1, name="qae_phase_reference")
    qae = build_minimal_coherent_qae_circuit(amplitude_unitary, phase_bits)
    circuit.compose(qae, qubits=[*phase_qubits, system_qubit], inplace=True)
    circuit.h(target_qubit)
    _apply_phase_conditioned_z_evolution(
        circuit, phase_qubits, target_qubit, gamma, magnitude_bits
    )
    return circuit


def qae_signed_downstream_validation(
    amplitude_unitary: np.ndarray,
    phase_bits: int = 4,
    magnitude_bits: int = 3,
    gamma: float = 0.7,
) -> dict[str, float]:
    """Validate decoder, downstream action, and signed-register uncomputation."""
    from qiskit.quantum_info import partial_trace

    actual_circuit = build_qae_signed_downstream_circuit(
        amplitude_unitary, phase_bits, magnitude_bits, gamma, uncompute_signed=True
    )
    reference_circuit = build_qae_phase_reference_downstream_circuit(
        amplitude_unitary, phase_bits, magnitude_bits, gamma
    )
    actual = Statevector.from_instruction(actual_circuit)
    reference = Statevector.from_instruction(reference_circuit)
    fidelity = float(state_fidelity(reference, actual))

    sign_qubit = phase_bits + 1
    magnitude_qubits = tuple(range(phase_bits + 2, phase_bits + 2 + magnitude_bits))
    signed_qubits = [sign_qubit, *magnitude_qubits]
    probs = actual.probabilities(qargs=signed_qubits)
    zero_probability = float(probs[0])
    reduced_signed = partial_trace(actual, [q for q in range(actual.num_qubits) if q not in signed_qubits])
    signed_purity = float(np.real(np.trace(reduced_signed.data @ reduced_signed.data)))
    return {
        "state_fidelity": fidelity,
        "signed_zero_probability": zero_probability,
        "signed_register_purity": signed_purity,
    }


def qae_signed_error_budget(
    amplitude_unitary: np.ndarray,
    phase_bits: int,
    magnitude_bits: int,
    gamma: float,
) -> dict[str, float]:
    """Decompose QAE, signed-decoder, and downstream errors.

    Errors are probability-weighted over the coherent QAE phase distribution.
    The downstream metric is the weighted spectral-norm error between
    ``exp(-i gamma s Z)`` and ``exp(-i gamma s_hat_y Z)``.  The Lipschitz
    upper bound ``|gamma| E[|s-s_hat_y|]`` is returned separately.
    """
    if phase_bits <= 0 or magnitude_bits <= 0:
        raise ValueError("phase_bits and magnitude_bits must be positive")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    summary = qae_amplitude_summary(amplitude_unitary, phase_bits)
    exact_p = float(summary["exact_probability"])
    exact_s = 2.0 * exact_p - 1.0
    distribution = summary["distribution"]

    qae_abs = 0.0
    decoder_abs = 0.0
    total_abs = 0.0
    downstream_norm = 0.0
    downstream_infidelity = 0.0
    for index, weight in distribution.items():
        phase_p = qae_amplitude_from_phase_index(index, phase_bits)
        phase_s = 2.0 * phase_p - 1.0
        _, _, decoded_s, _ = phase_index_to_signed_code(
            index, phase_bits, magnitude_bits
        )
        qae_abs += weight * abs(phase_s - exact_s)
        decoder_abs += weight * abs(decoded_s - phase_s)
        delta = decoded_s - exact_s
        total_abs += weight * abs(delta)
        op_norm = 2.0 * abs(np.sin(0.5 * float(gamma) * delta))
        downstream_norm += weight * op_norm
        # For target |+>, overlap squared after two Z rotations.
        downstream_infidelity += weight * (np.sin(float(gamma) * delta) ** 2)

    return {
        "exact_probability": exact_p,
        "exact_signed_value": exact_s,
        "expected_qae_signed_abs_error": float(qae_abs),
        "expected_decoder_abs_error": float(decoder_abs),
        "expected_total_signed_abs_error": float(total_abs),
        "expected_downstream_operator_norm_error": float(downstream_norm),
        "downstream_lipschitz_bound": float(abs(gamma) * total_abs),
        "expected_downstream_state_infidelity": float(downstream_infidelity),
    }
