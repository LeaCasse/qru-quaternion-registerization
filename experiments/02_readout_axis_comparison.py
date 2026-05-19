from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import (
    quaternion_guided_axis,
    quaternion_weighted_energy,
    hidden_motion_ratio,
    readout_values,
)


def main(seed: int = 13, depth: int = 5, n_x: int = 161) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    Q = paths["quaternions"]

    axes = {
        "e_x": np.array([1.0, 0.0, 0.0]),
        "e_y": np.array([0.0, 1.0, 0.0]),
        "e_z": np.array([0.0, 0.0, 1.0]),
        "v_quat": quaternion_guided_axis(R, Q, xs),
    }
    rows = []
    for name, v in axes.items():
        rows.append({
            "axis": name,
            "v_x": float(v[0]),
            "v_y": float(v[1]),
            "v_z": float(v[2]),
            "quat_weighted_energy": quaternion_weighted_energy(R, Q, v, xs),
            "hidden_motion_ratio": hidden_motion_ratio(R, Q, v),
        })
    with (TAB / "02_readout_axis_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["axis"] for r in rows], [r["quat_weighted_energy"] for r in rows])
    ax.set_ylabel(r"$E_{quat}(v)$")
    ax.set_title("Quaternion-weighted readout energy")
    fig.tight_layout()
    fig.savefig(FIG / "02_quaternion_weighted_energy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ["e_x", "e_y", "e_z", "v_quat"]:
        ax.plot(xs, readout_values(R, axes[name]), label=name)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$s_v(x)$")
    ax.legend(fontsize=8)
    ax.set_title("Signed readouts compared")
    fig.tight_layout()
    fig.savefig(FIG / "02_signed_readouts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
