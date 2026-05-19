from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import (
    quaternion_guided_axis,
    quaternion_motion,
    readout_variations,
    local_hidden_motion_ratio,
    hidden_motion_ratio,
)


def main(seed: int = 13, depth: int = 5, n_x: int = 161) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    Q = paths["quaternions"]

    ez = np.array([0.0, 0.0, 1.0])
    vq = quaternion_guided_axis(R, Q, xs)

    dq = quaternion_motion(Q)
    dz = readout_variations(R, ez)
    dvq = readout_variations(R, vq)
    Hz = local_hidden_motion_ratio(R, Q, ez)
    Hvq = local_hidden_motion_ratio(R, Q, vq)
    x_mid = 0.5 * (xs[:-1] + xs[1:])

    rows = []
    for i in range(len(dq)):
        rows.append({
            "x_mid": float(x_mid[i]),
            "delta_q": float(dq[i]),
            "delta_z": float(dz[i]),
            "delta_vquat": float(dvq[i]),
            "hidden_z": float(Hz[i]),
            "hidden_vquat": float(Hvq[i]),
        })
    with (TAB / "01_hidden_motion_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = [
        {"axis": "e_z", "global_hidden_motion_ratio": hidden_motion_ratio(R, Q, ez)},
        {"axis": "v_quat", "global_hidden_motion_ratio": hidden_motion_ratio(R, Q, vq)},
    ]
    with (TAB / "01_hidden_motion_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(x_mid, dq, color="black", label=r"$\delta_q=d_{S^3}(q_i,q_{i+1})$")
    axs[0].set_ylabel("quaternion motion")
    axs[0].legend(fontsize=8)
    axs[1].plot(x_mid, dz, label=r"$\delta_{e_z}$")
    axs[1].plot(x_mid, dvq, label=r"$\delta_{v_{quat}}$")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("visible readout variation")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_hidden_motion_diagnostics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x_mid, Hz, label=r"$H_{e_z}$")
    ax.plot(x_mid, Hvq, label=r"$H_{v_{quat}}$")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("local hidden-motion ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_local_hidden_motion_ratio.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
