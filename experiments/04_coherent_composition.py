from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile

from _common import FIG, TAB
from qru_registerization.transpile_config import PAPER_TRANSPILE_KWARGS
from qru_registerization.coherent_register import (
    build_register_controlled_z_circuit,
    build_two_branch_pipeline_segments,
    register_controlled_z_fidelity,
    two_branch_coherence_echo_metrics,
)
from qru_registerization.fixed_point import decode_signed_fixed_point
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import (
    quotient_quaternion_weighted_axis,
    readout_values,
)


def _register_validation() -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    gamma_values = (-0.73, 0.19, 1.17)
    for m in (2, 3):
        for gamma in gamma_values:
            for sign_bit in (0, 1):
                for bits_tuple in itertools.product((0, 1), repeat=m):
                    bits = np.asarray(bits_tuple, dtype=int)
                    value = decode_signed_fixed_point(sign_bit, bits)
                    circuit = build_register_controlled_z_circuit(sign_bit, bits, gamma)
                    compiled = transpile(
                        circuit,
                        **PAPER_TRANSPILE_KWARGS,
                    )
                    fidelity = register_controlled_z_fidelity(sign_bit, bits, gamma)
                    rows.append(
                        {
                            "m": m,
                            "gamma": gamma,
                            "sign_bit": sign_bit,
                            "magnitude_integer": int("".join(str(int(b)) for b in bits), 2),
                            "signed_value": value,
                            "fidelity": fidelity,
                            "infidelity": max(0.0, 1.0 - fidelity),
                            "logical_depth": circuit.depth(),
                            "compiled_depth": compiled.depth(),
                            "compiled_cx_count": int(compiled.count_ops().get("cx", 0)),
                        }
                    )

    summary: list[dict[str, float | int]] = []
    for m in (2, 3):
        selected = [row for row in rows if row["m"] == m]
        summary.append(
            {
                "m": m,
                "tested_circuits": len(selected),
                "minimum_fidelity": min(float(row["fidelity"]) for row in selected),
                "maximum_infidelity": max(float(row["infidelity"]) for row in selected),
                "maximum_compiled_depth": max(int(row["compiled_depth"]) for row in selected),
                "maximum_compiled_cx_count": max(int(row["compiled_cx_count"]) for row in selected),
            }
        )
    return rows, summary


def _coherent_demo() -> tuple[dict[str, object], QuantumCircuit]:
    params = random_qru_params(depth=3, seed=23, scale=0.8)
    xs = np.linspace(-np.pi, np.pi, 161)
    paths = compute_paths(params, xs)
    axis_result = quotient_quaternion_weighted_axis(
        paths["bloch_direct"], paths["states"], xs=xs
    )
    if axis_result.axis is None:
        raise RuntimeError("coherent demo trajectory did not define an identifiable axis")
    signed = readout_values(paths["bloch_direct"], axis_result.axis)
    i_zero = int(np.argmin(signed))
    i_one = int(np.argmax(signed))
    x_zero = float(xs[i_zero])
    x_one = float(xs[i_one])

    metrics = two_branch_coherence_echo_metrics(
        params=params,
        x_zero=x_zero,
        x_one=x_one,
        axis=axis_result.axis,
        m=3,
        gamma=0.73,
    )
    preparation, echo_tail, _ = build_two_branch_pipeline_segments(
        params, x_zero, x_one, axis_result.axis, 3, 0.73
    )
    full_echo = preparation.compose(echo_tail)
    compiled = transpile(
        full_echo,
        **PAPER_TRANSPILE_KWARGS,
    )
    metrics = {
        **metrics,
        "x_zero": x_zero,
        "x_one": x_one,
        "axis_x": float(axis_result.axis[0]),
        "axis_y": float(axis_result.axis[1]),
        "axis_z": float(axis_result.axis[2]),
        "axis_eigengap": float(axis_result.eigengap),
        "qubit_count": full_echo.num_qubits,
        "logical_depth": full_echo.depth(),
        "compiled_depth": compiled.depth(),
        "compiled_cx_count": int(compiled.count_ops().get("cx", 0)),
    }
    return metrics, full_echo


def main() -> None:
    rows, summary = _register_validation()
    coherent, _ = _coherent_demo()

    with (TAB / "04_register_controlled_z_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (TAB / "04_register_controlled_z_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    coherent_row = {
        key: value
        for key, value in coherent.items()
        if not isinstance(value, (tuple, list, np.ndarray))
    }
    coherent_row["signed_value_zero"] = float(coherent["signed_values"][0])
    coherent_row["signed_value_one"] = float(coherent["signed_values"][1])
    coherent_row["quantized_value_zero"] = float(coherent["quantized_values"][0])
    coherent_row["quantized_value_one"] = float(coherent["quantized_values"][1])
    coherent_row["code_zero"] = str(coherent["codes"][0])
    coherent_row["code_one"] = str(coherent["codes"][1])
    with (TAB / "04_two_branch_coherence_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coherent_row.keys()))
        writer.writeheader()
        writer.writerow(coherent_row)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    gamma_plot = 1.17
    for m, marker in ((2, "o"), (3, "s")):
        selected = [
            row for row in rows
            if row["m"] == m and np.isclose(float(row["gamma"]), gamma_plot)
        ]
        selected.sort(key=lambda row: float(row["signed_value"]))
        axes[0].plot(
            [float(row["signed_value"]) for row in selected],
            [float(row["infidelity"]) + np.finfo(float).eps for row in selected],
            marker=marker,
            linestyle="none",
            label=f"m={m}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Signed register value")
    axes[0].set_ylabel("Statevector infidelity")
    axes[0].set_title(r"Register-controlled $e^{-i\gamma s Z}$")
    axes[0].legend()

    labels = [r"$x_0$", r"$x_1$"]
    exact = [float(v) for v in coherent["signed_values"]]
    quantized = [float(v) for v in coherent["quantized_values"]]
    positions = np.arange(2)
    axes[1].bar(positions - 0.17, exact, width=0.34, label="Exact readout")
    axes[1].bar(positions + 0.17, quantized, width=0.34, label="3-bit register")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Signed value")
    axes[1].set_title("Two QRU branches")
    axes[1].legend()

    axes[2].bar(
        ["Coherent", "Dephased"],
        [
            float(coherent["p_input_zero_coherent"]),
            float(coherent["p_input_zero_after_dephasing"]),
        ],
    )
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel(r"Echo probability $P(0)$")
    axes[2].set_title("Interference echo")

    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    fig.savefig(FIG / "04_register_controlled_z_validation.pdf")
    plt.close(fig)

    print("Register-controlled Z validation:")
    for row in summary:
        print(row)
    print("Two-branch coherence demo:")
    print(coherent_row)


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
