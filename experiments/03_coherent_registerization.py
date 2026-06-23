from __future__ import annotations

import csv
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile

from _common import FIG, TAB
from qru_registerization.transpile_config import PAPER_TRANSPILE_KWARGS
from qru_registerization.amplitude_interface import (
    basis_rotation_from_axis,
    basis_rotation_probability,
    directional_observable,
    projector_probability,
)
from qru_registerization.bloch import Z
from qru_registerization.coherent_register import (
    build_minimal_coherent_qae_circuit,
    build_phase_to_signed_decoder_circuit,
    build_qae_signed_downstream_circuit,
    build_signed_downstream_block_circuit,
    directional_amplitude_unitary,
    qae_amplitude_summary,
    qae_phase_distribution_from_probability,
    phase_index_to_signed_code,
    qae_signed_downstream_validation,
    qae_signed_error_budget,
    qae_signed_error_budget_from_probability,
)
from qru_registerization.fixed_point import max_quantization_error_bound
from qru_registerization.gates import qru_unitary, random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.quaternion_diagnostics import quotient_quaternion_weighted_axis


def _count_resources(circuit):
    compiled = transpile(circuit, **PAPER_TRANSPILE_KWARGS)
    logical_ops = circuit.count_ops()
    compiled_ops = compiled.count_ops()
    return {
        "num_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "compiled_depth": compiled.depth(),
        "compiled_cx_count": int(compiled_ops.get("cx", 0)),
        "logical_mcx_count": int(logical_ops.get("mcx", 0)),
        "logical_mcrz_count": int(logical_ops.get("mcrz", 0)),
    }


def _write_latex_table(path, rows):
    selected = [
        r for r in rows
        if r["block"] in {"qae_qpe", "phase_to_signed_lookup", "signed_downstream", "full_pipeline"}
    ]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$(m_\phi,m_s)$ & block & qubits & logical depth & transpiled depth & CX \\",
        r"\midrule",
    ]
    for r in selected:
        label = f"({r['phase_bits']},{r['magnitude_bits']})"
        block = str(r["block"]).replace("_", r"\_")
        lines.append(
            f"{label} & {block} & {r['num_qubits']} & {r['logical_depth']} & "
            f"{r['compiled_depth']} & {r['compiled_cx_count']} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(seed: int = 13, depth: int = 5, n_x: int = 161, full_resources: bool = False) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    states = paths["states"]
    axis_result = quotient_quaternion_weighted_axis(R, states, xs=xs)
    if axis_result.axis is None:
        raise RuntimeError("registerization experiment requires an identifiable axis")
    axis = axis_result.axis

    rows = []
    for m in range(2, 13):
        reg = registerize_path(states, R, axis, m=m)
        err = np.abs(reg["s_quant"] - reg["s"])
        rows.append({
            "m_bits": m,
            "max_abs_error": float(np.max(err)),
            "mean_abs_error": float(np.mean(err)),
            "mse": float(np.mean(err**2)),
            "bound_2_minus_m": max_quantization_error_bound(m),
        })
    with (TAB / "03_registerization_precision.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(31)
    validation_rows = []
    for index in range(100):
        test_axis = rng.normal(size=3)
        B = basis_rotation_from_axis(test_axis)
        observable_error = np.linalg.norm(
            B.conj().T @ Z @ B - directional_observable(test_axis), ord=2
        )
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi /= np.linalg.norm(psi)
        probability_error = abs(
            basis_rotation_probability(psi, test_axis)
            - projector_probability(psi, test_axis)
        )
        validation_rows.append({
            "sample": index,
            "observable_operator_error": float(observable_error),
            "probability_error": float(probability_error),
        })
    with (TAB / "03_basis_rotation_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    signed_values = R @ axis
    probabilities = 0.5 * (1.0 + signed_values)
    qae_index = int(np.argmin(np.abs(probabilities - 0.30)))
    qae_x = float(xs[qae_index])
    amplitude_unitary = directional_amplitude_unitary(
        qru_unitary(params, qae_x), axis
    )
    qae_summaries = {}
    qae_rows = []
    for phase_bits in (2, 3, 4, 5):
        qae_summary = qae_amplitude_summary(amplitude_unitary, phase_bits=phase_bits)
        qae_summaries[phase_bits] = qae_summary
        qae_circuit = build_minimal_coherent_qae_circuit(
            amplitude_unitary, phase_bits=phase_bits
        )
        qae_compiled = transpile(
            qae_circuit,
            **PAPER_TRANSPILE_KWARGS,
        )
        for phase_index, probability in qae_summary["distribution"].items():
            qae_rows.append({
                "phase_bits": phase_bits,
                "phase_index": phase_index,
                "phase_fraction": phase_index / float(2**phase_bits),
                "phase_probability": probability,
                "decoded_amplitude": qae_summary["amplitude_estimates"][phase_index],
                "exact_amplitude": qae_summary["exact_probability"],
                "mean_amplitude_estimate": qae_summary["mean_estimate"],
                "absolute_mean_error": qae_summary["absolute_mean_error"],
                "mode_amplitude_estimate": qae_summary["mode_estimate"],
                "x": qae_x,
                "logical_depth": qae_circuit.depth(),
                "compiled_depth": qae_compiled.depth(),
                "compiled_cx_count": int(qae_compiled.count_ops().get("cx", 0)),
            })
    with (TAB / "03_coherent_qae_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qae_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qae_rows)

    decoder_rows = []
    phase_bits = 4
    magnitude_bits = 3
    for phase_index in range(2**phase_bits):
        sign_bit, bits, decoded_signed, exact_signed = phase_index_to_signed_code(
            phase_index, phase_bits, magnitude_bits
        )
        decoder_rows.append({
            "phase_index": phase_index,
            "phase_fraction": phase_index / float(2**phase_bits),
            "exact_signed_value": exact_signed,
            "sign_bit": sign_bit,
            "magnitude_bits": "".join(str(int(bit)) for bit in bits),
            "decoded_signed_value": decoded_signed,
            "absolute_decoder_error": abs(decoded_signed - exact_signed),
        })
    with (TAB / "03_phase_to_signed_decoder.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decoder_rows[0].keys()))
        writer.writeheader()
        writer.writerows(decoder_rows)

    downstream_gamma = 0.73
    integrated = qae_signed_downstream_validation(
        amplitude_unitary,
        phase_bits=phase_bits,
        magnitude_bits=magnitude_bits,
        gamma=downstream_gamma,
    )
    integrated_circuit = build_qae_signed_downstream_circuit(
        amplitude_unitary,
        phase_bits=phase_bits,
        magnitude_bits=magnitude_bits,
        gamma=downstream_gamma,
    )
    integrated_compiled = transpile(
        integrated_circuit,
        **PAPER_TRANSPILE_KWARGS,
    )
    integrated_row = {
        "phase_bits": phase_bits,
        "magnitude_bits": magnitude_bits,
        "gamma": downstream_gamma,
        "num_qubits": integrated_circuit.num_qubits,
        "logical_depth": integrated_circuit.depth(),
        "compiled_depth": integrated_compiled.depth(),
        "compiled_cx_count": int(integrated_compiled.count_ops().get("cx", 0)),
        **integrated,
    }
    with (TAB / "03_qae_signed_downstream_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(integrated_row.keys()))
        writer.writeheader()
        writer.writerow(integrated_row)

    # Precision/resource tradeoff.  The full-resource mode is used by run_all
    # so paper outputs contain regenerated statevector validations, not placeholders.
    tradeoff_rows = []
    breakdown_rows = []
    precision_configs = ((3, 2), (4, 3), (5, 4))
    for phase_bits_cfg, magnitude_bits_cfg in precision_configs:
        budget = qae_signed_error_budget_from_probability(
            float(abs(amplitude_unitary[1, 0]) ** 2),
            phase_bits_cfg,
            magnitude_bits_cfg,
            downstream_gamma,
        )
        full_circuit = build_qae_signed_downstream_circuit(
            amplitude_unitary,
            phase_bits=phase_bits_cfg,
            magnitude_bits=magnitude_bits_cfg,
            gamma=downstream_gamma,
        )
        full_resources = _count_resources(full_circuit)
        validation = qae_signed_downstream_validation(
            amplitude_unitary,
            phase_bits=phase_bits_cfg,
            magnitude_bits=magnitude_bits_cfg,
            gamma=downstream_gamma,
        )
        tradeoff_rows.append({
            "phase_bits": phase_bits_cfg,
            "magnitude_bits": magnitude_bits_cfg,
            **full_resources,
            **budget,
            **validation,
        })

        block_circuits = {
            "qae_qpe": build_minimal_coherent_qae_circuit(
                amplitude_unitary, phase_bits=phase_bits_cfg
            ),
            "phase_to_signed_lookup": build_phase_to_signed_decoder_circuit(
                phase_bits_cfg, magnitude_bits_cfg
            ),
            "signed_downstream": build_signed_downstream_block_circuit(
                magnitude_bits_cfg, downstream_gamma
            ),
            "uncompute_signed_lookup": build_phase_to_signed_decoder_circuit(
                phase_bits_cfg, magnitude_bits_cfg
            ).inverse(),
            "full_pipeline": full_circuit,
        }
        for block, circuit in block_circuits.items():
            breakdown_rows.append({
                "phase_bits": phase_bits_cfg,
                "magnitude_bits": magnitude_bits_cfg,
                "block": block,
                **_count_resources(circuit),
            })

    with (TAB / "03_precision_resource_tradeoff.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tradeoff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tradeoff_rows)

    with (TAB / "coherent_resource_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
        writer.writeheader()
        writer.writerows(breakdown_rows)
    _write_latex_table(TAB / "coherent_resource_breakdown.tex", breakdown_rows)

    robustness_configs = ((3, 2), (4, 3), (5, 4))
    probability_grid = np.linspace(0.0, 1.0, 201)
    robustness_rows = []
    for probability in probability_grid:
        for phase_bits_cfg, magnitude_bits_cfg in robustness_configs:
            budget = qae_signed_error_budget_from_probability(
                float(probability),
                phase_bits_cfg,
                magnitude_bits_cfg,
                downstream_gamma,
            )
            robustness_rows.append({
                "source": "analytic_grid",
                "depth": "",
                "seed": "",
                "x": "",
                "phase_bits": phase_bits_cfg,
                "magnitude_bits": magnitude_bits_cfg,
                **budget,
            })

    qru_rows = []
    sample_xs = np.linspace(-np.pi, np.pi, 31)
    for sample_depth in (2, 3, 5, 8):
        for sample_seed in range(12):
            sample_params = random_qru_params(
                depth=sample_depth, seed=sample_seed, scale=0.9
            )
            sample_paths = compute_paths(sample_params, sample_xs)
            sample_axis_result = quotient_quaternion_weighted_axis(
                sample_paths["bloch_direct"], sample_paths["states"], xs=sample_xs
            )
            if sample_axis_result.axis is None:
                continue
            sample_probabilities = 0.5 * (
                1.0 + sample_paths["bloch_direct"] @ sample_axis_result.axis
            )
            for sample_x, probability in zip(sample_xs, sample_probabilities, strict=True):
                for phase_bits_cfg, magnitude_bits_cfg in robustness_configs:
                    budget = qae_signed_error_budget_from_probability(
                        float(np.clip(probability, 0.0, 1.0)),
                        phase_bits_cfg,
                        magnitude_bits_cfg,
                        downstream_gamma,
                    )
                    row = {
                        "source": "qru",
                        "depth": sample_depth,
                        "seed": sample_seed,
                        "x": float(sample_x),
                        "phase_bits": phase_bits_cfg,
                        "magnitude_bits": magnitude_bits_cfg,
                        **budget,
                    }
                    qru_rows.append(row)
                    robustness_rows.append(row)

    with (TAB / "03_probability_robustness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(robustness_rows[0].keys()))
        writer.writeheader()
        writer.writerows(robustness_rows)

    summary_rows = []
    for source, source_rows in (("analytic_grid", [r for r in robustness_rows if r["source"] == "analytic_grid"]), ("qru", qru_rows)):
        for phase_bits_cfg, magnitude_bits_cfg in robustness_configs:
            selected = [
                r for r in source_rows
                if r["phase_bits"] == phase_bits_cfg
                and r["magnitude_bits"] == magnitude_bits_cfg
            ]
            total_errors = np.array([r["expected_total_signed_abs_error"] for r in selected])
            downstream_errors = np.array([max(r["expected_downstream_state_infidelity"], np.finfo(float).eps) for r in selected])
            summary_rows.append({
                "source": source,
                "phase_bits": phase_bits_cfg,
                "magnitude_bits": magnitude_bits_cfg,
                "count": len(selected),
                "mean_total_signed_abs_error": float(np.mean(total_errors)),
                "median_total_signed_abs_error": float(np.median(total_errors)),
                "p95_total_signed_abs_error": float(np.quantile(total_errors, 0.95)),
                "max_total_signed_abs_error": float(np.max(total_errors)),
                "fraction_total_error_le_0_02": float(np.mean(total_errors <= 0.02)),
                "fraction_total_error_le_0_05": float(np.mean(total_errors <= 0.05)),
                "mean_downstream_infidelity": float(np.mean(downstream_errors)),
                "p95_downstream_infidelity": float(np.quantile(downstream_errors, 0.95)),
                "max_downstream_infidelity": float(np.max(downstream_errors)),
            })
    with (TAB / "03_probability_robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    downstream_bound_rows = []
    for probability in np.linspace(0.0, 1.0, 201):
        exact_s = 2.0 * float(probability) - 1.0
        for phase_bits_cfg, magnitude_bits_cfg in robustness_configs:
            distribution = qae_phase_distribution_from_probability(float(probability), phase_bits_cfg)
            weighted_deltas = []
            total_abs = 0.0
            for index, weight in distribution.items():
                _, _, decoded_s, _ = phase_index_to_signed_code(
                    index, phase_bits_cfg, magnitude_bits_cfg
                )
                delta = decoded_s - exact_s
                weighted_deltas.append((weight, delta))
                total_abs += weight * abs(delta)
            for gamma in (0.1, 0.5, 0.73, 1.0, 1.5):
                downstream_norm = sum(
                    weight * 2.0 * abs(np.sin(0.5 * float(gamma) * delta))
                    for weight, delta in weighted_deltas
                )
                lipschitz_bound = abs(float(gamma)) * total_abs
                downstream_bound_rows.append({
                    "phase_bits": phase_bits_cfg,
                    "magnitude_bits": magnitude_bits_cfg,
                    "gamma": gamma,
                    "probability": float(probability),
                    "expected_total_signed_abs_error": float(total_abs),
                    "expected_downstream_operator_norm_error": float(downstream_norm),
                    "downstream_lipschitz_bound": float(lipschitz_bound),
                    "bound_slack": float(lipschitz_bound - downstream_norm),
                })
    with (TAB / "downstream_bound_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(downstream_bound_rows[0].keys()))
        writer.writeheader()
        writer.writerows(downstream_bound_rows)
    del downstream_bound_rows
    gc.collect()

    # Reload plotting inputs from disk.  This keeps figure generation tied to
    # persisted paper tables and avoids backend stalls from large live objects.
    with (TAB / "03_probability_robustness.csv").open(encoding="utf-8") as f:
        robustness_rows = list(csv.DictReader(f))
    for row in robustness_rows:
        for key in (
            "phase_bits",
            "magnitude_bits",
            "exact_probability",
            "expected_total_signed_abs_error",
            "expected_downstream_state_infidelity",
        ):
            if row[key] != "":
                row[key] = float(row[key])
    with (TAB / "03_probability_robustness_summary.csv").open(encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    for row in summary_rows:
        for key in (
            "phase_bits",
            "magnitude_bits",
            "mean_total_signed_abs_error",
            "fraction_total_error_le_0_02",
        ):
            row[key] = float(row[key])
    gc.collect()

    phase_bit_values = np.array(sorted(qae_summaries))
    print("Coherent QAE precision validation:")
    for phase_bits in phase_bit_values:
        summary = qae_summaries[int(phase_bits)]
        selected = [row for row in qae_rows if row["phase_bits"] == phase_bits]
        print({
            "phase_bits": int(phase_bits),
            "x": qae_x,
            "exact_probability": summary["exact_probability"],
            "mean_estimate": summary["mean_estimate"],
            "absolute_mean_error": summary["absolute_mean_error"],
            "mode_estimate": summary["mode_estimate"],
            "compiled_depth": selected[0]["compiled_depth"],
            "compiled_cx_count": selected[0]["compiled_cx_count"],
        })
    print({"integrated_qae_signed_downstream": integrated_row})
    print("Precision-resource tradeoff:")
    for row in tradeoff_rows:
        print(row)
    print("Probability robustness summary:")
    for row in summary_rows:
        print(row)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-resources",
        action="store_true",
        help="retained for compatibility; paper runs now regenerate full resources by default",
    )
    args = parser.parse_args()
    main(full_resources=args.full_resources)
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
