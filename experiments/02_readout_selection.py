from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.bloch import bloch_vector
from qru_registerization.gates import ry, rz
from qru_registerization.quaternion_diagnostics import (
    best_pauli_axis,
    quotient_quaternion_weighted_axis,
    raw_quaternion_weighted_axis,
    readout_values,
    state_motion,
    state_motion_weighted_axis,
    unweighted_bloch_pca_axis,
    weighted_tangent_energy,
)
from qru_registerization.quaternion_geometry import su2_to_quaternion


def _row(name: str, result, R: np.ndarray, weights: np.ndarray, xs: np.ndarray) -> dict[str, float | str | bool]:
    axis = result.axis
    if axis is None:
        return {
            "method": name,
            "identifiable": False,
            "v_x": np.nan,
            "v_y": np.nan,
            "v_z": np.nan,
            "eigengap": result.eigengap,
            "lambda_1": result.eigenvalues[0],
            "lambda_2": result.eigenvalues[1],
            "lambda_3": result.eigenvalues[2],
            "quotient_weighted_energy": np.nan,
        }
    return {
        "method": name,
        "identifiable": True,
        "v_x": float(axis[0]),
        "v_y": float(axis[1]),
        "v_z": float(axis[2]),
        "eigengap": float(result.eigengap),
        "lambda_1": float(result.eigenvalues[0]),
        "lambda_2": float(result.eigenvalues[1]),
        "lambda_3": float(result.eigenvalues[2]),
        "quotient_weighted_energy": weighted_tangent_energy(R, axis, weights=weights, xs=xs),
    }


def main(n_x: int = 161, c: float = 0.35) -> None:
    """Physical projection blindness: Z is constant while the state moves on a Bloch arc."""
    xs = np.linspace(-np.pi / 3.0, np.pi / 3.0, n_x)
    theta = float(np.arccos(c))
    ket0 = np.array([1.0, 0.0], dtype=complex)

    unitaries = np.asarray([rz(float(x)) @ ry(theta) for x in xs])
    states = np.asarray([U @ ket0 for U in unitaries])
    R = np.asarray([bloch_vector(psi) for psi in states])
    Q = np.asarray([su2_to_quaternion(U) for U in unitaries])
    quotient_weights = state_motion(states) ** 2

    methods = {
        "z_axis": type("FixedResult", (), {
            "axis": np.array([0.0, 0.0, 1.0]),
            "eigengap": np.nan,
            "eigenvalues": np.array([np.nan, np.nan, np.nan]),
        })(),
        "best_pauli": best_pauli_axis(R, weights=quotient_weights, xs=xs),
        "unweighted_bloch_pca": unweighted_bloch_pca_axis(R, xs=xs),
        "state_motion_weighted": state_motion_weighted_axis(R, states, xs=xs),
        "raw_quaternion_weighted": raw_quaternion_weighted_axis(R, Q, xs=xs),
        "quotient_quaternion_weighted": quotient_quaternion_weighted_axis(R, states, xs=xs),
    }

    assert float(np.ptp(R[:, 2])) < 1e-12
    for name in ["unweighted_bloch_pca", "state_motion_weighted", "raw_quaternion_weighted", "quotient_quaternion_weighted"]:
        result = methods[name]
        assert result.identifiable
        assert result.axis is not None
        assert abs(float(result.axis[2])) < 1e-10

    rows = [_row(name, result, R, quotient_weights, xs) for name, result in methods.items()]
    with (TAB / "02_projection_blindness_axes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(xs, R[:, 0], label=r"$r_x$")
    axes[0].plot(xs, R[:, 1], label=r"$r_y$")
    axes[0].plot(xs, R[:, 2], label=r"$r_z$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Bloch coordinate")
    axes[0].set_title(r"Observable state motion with constant $\langle Z\rangle$")
    axes[0].legend(fontsize=8)

    selected = ["z_axis", "unweighted_bloch_pca", "quotient_quaternion_weighted"]
    for name in selected:
        axis = methods[name].axis
        axes[1].plot(xs, readout_values(R, axis), label=name)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$s_v(x)$")
    axes[1].set_title("Fixed and geometry-selected signed readouts")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "02_projection_blindness.pdf")
    fig.savefig(FIG / "02_projection_blindness.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
