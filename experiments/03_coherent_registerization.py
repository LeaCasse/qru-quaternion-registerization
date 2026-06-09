from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile

from _common import FIG, TAB
from qru_registerization.amplitude_interface import (
    basis_rotation_from_axis,
    basis_rotation_probability,
    directional_observable,
    projector_probability,
)
from qru_registerization.bloch import Z
from qru_registerization.coherent_register import (
    build_minimal_coherent_qae_circuit,
    directional_amplitude_unitary,
    qae_amplitude_summary,
    phase_index_to_signed_code,
    qae_signed_downstream_validation,
    build_qae_signed_downstream_circuit,
    qae_signed_error_budget,
    qae_signed_error_budget_from_probability,
)
from qru_registerization.fixed_point import max_quantization_error_bound
from qru_registerization.gates import qru_unitary, random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.quaternion_diagnostics import quotient_quaternion_weighted_axis


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
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
            seed_transpiler=17,
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
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
        seed_transpiler=17,
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

    # Full transpilation of the finite lookup grows rapidly. The default run
    # uses resource counts previously validated with optimization_level=0;
    # pass --full-resources to recompute them.
    reference_resources = {
        (3, 2): {"num_qubits": 8, "logical_depth": 89, "compiled_depth": 795, "compiled_cx_count": 326},
        (4, 3): {"num_qubits": 10, "logical_depth": 272, "compiled_depth": 6156, "compiled_cx_count": 2170},
        (5, 4): {"num_qubits": 12, "logical_depth": 606, "compiled_depth": 19443, "compiled_cx_count": 6513},
    }
    tradeoff_rows = []
    for phase_bits_cfg, magnitude_bits_cfg in ((3, 2), (4, 3), (5, 4)):
        budget = qae_signed_error_budget_from_probability(
            float(abs(amplitude_unitary[1, 0]) ** 2),
            phase_bits_cfg,
            magnitude_bits_cfg,
            downstream_gamma,
        )
        resources = dict(reference_resources[(phase_bits_cfg, magnitude_bits_cfg)])
        validation = {
            "state_fidelity": float("nan"),
            "signed_zero_probability": float("nan"),
            "signed_register_purity": float("nan"),
        }
        if full_resources:
            cfg_circuit = build_qae_signed_downstream_circuit(
                amplitude_unitary,
                phase_bits=phase_bits_cfg,
                magnitude_bits=magnitude_bits_cfg,
                gamma=downstream_gamma,
            )
            cfg_compiled = transpile(
                cfg_circuit,
                basis_gates=["rz", "sx", "x", "cx"],
                optimization_level=0,
                seed_transpiler=17,
            )
            resources = {
                "num_qubits": cfg_circuit.num_qubits,
                "logical_depth": cfg_circuit.depth(),
                "compiled_depth": cfg_compiled.depth(),
                "compiled_cx_count": int(cfg_compiled.count_ops().get("cx", 0)),
            }
            validation = qae_signed_downstream_validation(
                amplitude_unitary,
                phase_bits=phase_bits_cfg,
                magnitude_bits=magnitude_bits_cfg,
                gamma=downstream_gamma,
            )
        elif (phase_bits_cfg, magnitude_bits_cfg) == (4, 3):
            validation = integrated
        tradeoff_rows.append({
            "phase_bits": phase_bits_cfg,
            "magnitude_bits": magnitude_bits_cfg,
            **resources,
            **budget,
            **validation,
        })
    with (TAB / "03_precision_resource_tradeoff.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(tradeoff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tradeoff_rows)


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
            downstream_errors = np.array([r["expected_downstream_state_infidelity"] for r in selected])
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

    fig2, axes2 = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for phase_bits_cfg, magnitude_bits_cfg in robustness_configs:
        selected = [
            r for r in robustness_rows
            if r["source"] == "analytic_grid"
            and r["phase_bits"] == phase_bits_cfg
            and r["magnitude_bits"] == magnitude_bits_cfg
        ]
        label = rf"$({phase_bits_cfg},{magnitude_bits_cfg})$"
        axes2[0, 0].plot(
            [r["exact_probability"] for r in selected],
            [r["expected_total_signed_abs_error"] for r in selected],
            label=label,
        )
        axes2[0, 1].plot(
            [r["exact_probability"] for r in selected],
            [r["expected_downstream_state_infidelity"] for r in selected],
            label=label,
        )
    axes2[0, 0].set_xlabel("exact probability $p$")
    axes2[0, 0].set_ylabel("expected signed absolute error")
    axes2[0, 0].set_title("Error across the full probability interval")
    axes2[0, 0].legend(fontsize=8)
    axes2[0, 1].set_xlabel("exact probability $p$")
    axes2[0, 1].set_ylabel("expected downstream infidelity")
    axes2[0, 1].set_yscale("log")
    axes2[0, 1].set_title("Downstream impact across $p$")
    axes2[0, 1].legend(fontsize=8)

    positions = np.arange(len(robustness_configs))
    labels_cfg = [f"({a},{b})" for a, b in robustness_configs]
    grid_summary = [r for r in summary_rows if r["source"] == "analytic_grid"]
    qru_summary = [r for r in summary_rows if r["source"] == "qru"]
    width = 0.36
    axes2[1, 0].bar(positions - width / 2, [r["mean_total_signed_abs_error"] for r in grid_summary], width, label="uniform p grid")
    axes2[1, 0].bar(positions + width / 2, [r["mean_total_signed_abs_error"] for r in qru_summary], width, label="QRU samples")
    axes2[1, 0].set_xticks(positions, labels_cfg)
    axes2[1, 0].set_xlabel(r"$(m_\phi,m_s)$")
    axes2[1, 0].set_ylabel("mean signed absolute error")
    axes2[1, 0].set_title("Uniform-grid and QRU error")
    axes2[1, 0].legend(fontsize=8)

    axes2[1, 1].bar(positions - width / 2, [r["fraction_total_error_le_0_02"] for r in grid_summary], width, label="uniform p grid")
    axes2[1, 1].bar(positions + width / 2, [r["fraction_total_error_le_0_02"] for r in qru_summary], width, label="QRU samples")
    axes2[1, 1].set_xticks(positions, labels_cfg)
    axes2[1, 1].set_ylim(0.0, 1.0)
    axes2[1, 1].set_xlabel(r"$(m_\phi,m_s)$")
    axes2[1, 1].set_ylabel(r"fraction with $\epsilon_s\leq0.02$")
    axes2[1, 1].set_title("Coverage of a 0.02 signed-error target")
    axes2[1, 1].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG / "03_probability_robustness.pdf")
    fig2.savefig(FIG / "03_probability_robustness.png", dpi=180)
    plt.close(fig2)

    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.2))
    axes = axes.ravel()
    m = np.array([r["m_bits"] for r in rows])
    axes[0].semilogy(m, [r["max_abs_error"] for r in rows], marker="o", label="max error")
    axes[0].semilogy(m, [r["bound_2_minus_m"] for r in rows], linestyle="--", label=r"$2^{-m}$ bound")
    axes[0].set_xlabel("magnitude bits $m$")
    axes[0].set_ylabel("absolute error")
    axes[0].set_title("Signed fixed-point quantization")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(
        [r["sample"] for r in validation_rows],
        [max(r["observable_operator_error"], np.finfo(float).eps) for r in validation_rows],
        label=r"$\|B_v^\dagger ZB_v-v\cdot\sigma\|_2$",
    )
    axes[1].semilogy(
        [r["sample"] for r in validation_rows],
        [max(r["probability_error"], np.finfo(float).eps) for r in validation_rows],
        label="probability mismatch",
    )
    axes[1].set_xlabel("random validation case")
    axes[1].set_ylabel("numerical residual")
    axes[1].set_title("Directional basis rotation")
    axes[1].legend(fontsize=8)

    phase_bit_values = np.array(sorted(qae_summaries))
    axes[2].semilogy(
        phase_bit_values,
        [qae_summaries[m]["absolute_mean_error"] for m in phase_bit_values],
        marker="o",
    )
    axes[2].set_xlabel("QAE phase bits")
    axes[2].set_ylabel(r"$|\mathbb{E}[\hat p]-p|$")
    axes[2].set_title(
        rf"Coherent QAE precision, exact $p={qae_summaries[2]['exact_probability']:.3f}$"
    )

    for phase_bits, marker in ((2, "o"), (4, "s")):
        summary = qae_summaries[phase_bits]
        aggregated = {}
        for index, weight in summary["distribution"].items():
            estimate = round(summary["amplitude_estimates"][index], 14)
            aggregated[estimate] = aggregated.get(estimate, 0.0) + weight
        estimates = sorted(aggregated)
        axes[3].plot(
            estimates,
            [aggregated[value] for value in estimates],
            marker=marker,
            label=f"phase bits={phase_bits}",
        )
    axes[3].axvline(
        qae_summaries[2]["exact_probability"],
        linestyle="--",
        label="exact probability",
    )
    axes[3].set_xlabel(r"decoded amplitude $\hat p$")
    axes[3].set_ylabel("probability mass")
    axes[3].set_title("Decoded coherent QAE distribution")
    axes[3].legend(fontsize=8)

    labels = [f"({r['phase_bits']},{r['magnitude_bits']})" for r in tradeoff_rows]
    x_positions = np.arange(len(tradeoff_rows))
    axes[4].plot(
        x_positions,
        [r["compiled_cx_count"] for r in tradeoff_rows],
        marker="o",
        label="CX count",
    )
    axes[4].plot(
        x_positions,
        [r["compiled_depth"] for r in tradeoff_rows],
        marker="s",
        label="compiled depth",
    )
    axes[4].set_xticks(x_positions, labels)
    axes[4].set_xlabel(r"$(m_\phi,m_s)$")
    axes[4].set_ylabel("compiled resources")
    axes[4].set_yscale("log")
    axes[4].set_title("Truth-table decoder resource growth")
    axes[4].legend(fontsize=8)

    axes[5].semilogy(
        x_positions,
        [r["expected_qae_signed_abs_error"] for r in tradeoff_rows],
        marker="o",
        label="QAE contribution",
    )
    axes[5].semilogy(
        x_positions,
        [r["expected_decoder_abs_error"] for r in tradeoff_rows],
        marker="s",
        label="decoder contribution",
    )
    axes[5].semilogy(
        x_positions,
        [r["expected_total_signed_abs_error"] for r in tradeoff_rows],
        marker="^",
        label="total signed error",
    )
    axes[5].set_xticks(x_positions, labels)
    axes[5].set_xlabel(r"$(m_\phi,m_s)$")
    axes[5].set_ylabel("expected absolute error")
    axes[5].set_title("Error decomposition")
    axes[5].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "03_registerization_precision.pdf")
    fig.savefig(FIG / "03_registerization_precision.png", dpi=180)
    plt.close(fig)

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
        help="recompute expensive transpilation and statevector validations for all precision pairs",
    )
    args = parser.parse_args()
    main(full_resources=args.full_resources)
