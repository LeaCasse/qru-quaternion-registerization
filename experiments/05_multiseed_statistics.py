from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis, quaternion_weighted_energy, hidden_motion_ratio


def main(depths=(2, 3, 5, 8), seeds=range(20), n_x: int = 121) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    ez = np.array([0.0, 0.0, 1.0])
    rows = []
    for depth in depths:
        for seed in seeds:
            params = random_qru_params(depth=depth, seed=int(seed), scale=0.9)
            paths = compute_paths(params, xs)
            R = paths["bloch_direct"]
            Q = paths["quaternions"]
            vq = quaternion_guided_axis(R, Q, xs)
            Ez = quaternion_weighted_energy(R, Q, ez, xs)
            Eq = quaternion_weighted_energy(R, Q, vq, xs)
            Hz = hidden_motion_ratio(R, Q, ez)
            Hq = hidden_motion_ratio(R, Q, vq)
            rows.append({
                "depth": int(depth),
                "seed": int(seed),
                "energy_ez": float(Ez),
                "energy_vquat": float(Eq),
                "energy_gain_vquat_over_ez": float(Eq / (Ez + 1e-12)),
                "hidden_ratio_ez": float(Hz),
                "hidden_ratio_vquat": float(Hq),
                "hidden_ratio_reduction": float(Hz / (Hq + 1e-12)),
            })
    with (TAB / "05_multiseed_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Per-depth summary.
    summary = []
    for depth in depths:
        rr = [r for r in rows if r["depth"] == depth]
        summary.append({
            "depth": int(depth),
            "median_energy_gain": float(np.median([r["energy_gain_vquat_over_ez"] for r in rr])),
            "mean_energy_gain": float(np.mean([r["energy_gain_vquat_over_ez"] for r in rr])),
            "median_hidden_ratio_reduction": float(np.median([r["hidden_ratio_reduction"] for r in rr])),
            "fraction_energy_gain_gt_1": float(np.mean([r["energy_gain_vquat_over_ez"] > 1.0 for r in rr])),
        })
    with (TAB / "05_multiseed_statistics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[r["energy_gain_vquat_over_ez"] for r in rows if r["depth"] == d] for d in depths]
    ax.boxplot(data, tick_labels=[str(d) for d in depths], showfliers=False)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("QRU depth")
    ax.set_ylabel(r"$E_{quat}(v_{quat})/E_{quat}(e_z)$")
    fig.tight_layout()
    fig.savefig(FIG / "05_multiseed_energy_gain.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
