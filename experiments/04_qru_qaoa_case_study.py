from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis, readout_values, hidden_motion_ratio, quaternion_weighted_energy
from qru_registerization.readout_estimators import shot_estimate_signed_coordinate, ideal_qae_register_estimate
from qru_registerization.qaoa_toy import optimize_qaoa_p1_grid, evaluate_qaoa_with_estimated_field


def main(seed: int = 13, depth: int = 5, n_x: int = 65, shots: int = 64, epsilon_p: float = 1/128, m_bits: int = 9, grid_size: int = 31) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    Q = paths["quaternions"]
    ez = np.array([0.0, 0.0, 1.0])
    vq = quaternion_guided_axis(R, Q, xs)

    h_true = readout_values(R, vq)
    z_readout = readout_values(R, ez)
    idxs = np.linspace(0, len(xs)-1, 17, dtype=int)
    rng = np.random.default_rng(seed + 2026)

    true_opts = {i: optimize_qaoa_p1_grid(float(h_true[i]), grid_size=grid_size) for i in idxs}
    rows = []
    for i in idxs:
        h = float(h_true[i])
        x = float(xs[i])

        h_est_z = shot_estimate_signed_coordinate(float(z_readout[i]), shots=shots, seed=int(rng.integers(1_000_000)))
        out_z = evaluate_qaoa_with_estimated_field(h, h_est_z, grid_size=grid_size, true_optimum=true_opts[i])
        rows.append({"x": x, "strategy": "measure_Z_reinject", "h_true": h, "h_est": h_est_z, **out_z})

        h_est_qae = ideal_qae_register_estimate(h, epsilon_p=epsilon_p, m_bits=m_bits, seed=int(rng.integers(1_000_000)))["s_quant"]
        out_qae = evaluate_qaoa_with_estimated_field(h, h_est_qae, grid_size=grid_size, true_optimum=true_opts[i])
        rows.append({"x": x, "strategy": "vquat_QAE_register", "h_true": h, "h_est": h_est_qae, **out_qae})

    for r in rows:
        r["abs_field_error"] = abs(r["h_est"] - r["h_true"])
        r["sign_error"] = float(np.sign(r["h_est"]) != np.sign(r["h_true"]))

    with (TAB / "04_qru_qaoa_case_study.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = []
    for strat in sorted(set(r["strategy"] for r in rows)):
        rr = [r for r in rows if r["strategy"] == strat]
        summary.append({
            "strategy": strat,
            "mean_abs_field_error": float(np.mean([r["abs_field_error"] for r in rr])),
            "mean_energy_gap_to_ground": float(np.mean([r["energy_gap_to_ground"] for r in rr])),
            "mean_ground_state_probability": float(np.mean([r["ground_state_probability"] for r in rr])),
            "sign_error_rate": float(np.mean([r["sign_error"] for r in rr])),
        })
    with (TAB / "04_qru_qaoa_case_study_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    diag = [
        {"axis": "e_z", "hidden_motion_ratio": hidden_motion_ratio(R, Q, ez), "quat_weighted_energy": quaternion_weighted_energy(R, Q, ez, xs)},
        {"axis": "v_quat", "hidden_motion_ratio": hidden_motion_ratio(R, Q, vq), "quat_weighted_energy": quaternion_weighted_energy(R, Q, vq, xs)},
    ]
    with (TAB / "04_qru_qaoa_axis_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag[0].keys()))
        writer.writeheader(); writer.writerows(diag)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, h_true, label=r"true $s_{v_{quat}}(x)$", linewidth=2)
    ax.plot(xs, z_readout, label=r"exact $s_z(x)$", linestyle="--")
    for strat, marker in [("measure_Z_reinject", "o"), ("vquat_QAE_register", "^")]:
        rr = [r for r in rows if r["strategy"] == strat]
        ax.scatter([r["x"] for r in rr], [r["h_est"] for r in rr], label=strat, marker=marker, s=30)
    ax.set_xlabel("x")
    ax.set_ylabel("downstream field h")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_qru_qaoa_field_estimates.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    labels = [s["strategy"] for s in summary]
    x = np.arange(len(labels))
    metrics = ["mean_abs_field_error", "mean_energy_gap_to_ground", "mean_ground_state_probability"]
    titles = ["Mean |field error|", "Mean energy gap", "Mean ground-state prob."]
    for ax, metric, title in zip(axs, metrics, titles):
        ax.bar(x, [s[metric] for s in summary])
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_qru_qaoa_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
