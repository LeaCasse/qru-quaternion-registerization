from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_diagnostics import (
    axis_angle,
    best_pauli_axis,
    observable_hidden_motion_ratio,
    quotient_quaternion_weighted_axis,
    raw_quaternion_weighted_axis,
    state_motion,
    state_motion_weighted_axis,
    unweighted_bloch_pca_axis,
    weighted_tangent_energy,
)


def _subpath(paths: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in paths.items() if key != "xs"} | {"xs": paths["xs"][mask]}


def _paired_bootstrap_ci(values: np.ndarray, *, seed: int = 2026, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main(depths=(2, 3, 5, 8), seeds=range(60), n_x: int = 181) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    train_mask = xs <= 0.0
    test_mask = xs >= 0.0
    rows: list[dict[str, float | int | str | bool]] = []

    for depth in depths:
        for seed in seeds:
            params = random_qru_params(depth=depth, seed=int(seed), scale=0.9)
            paths = compute_paths(params, xs)
            train = _subpath(paths, train_mask)
            test = _subpath(paths, test_mask)

            Rtr, Str, Qtr, Xtr = train["bloch_direct"], train["states"], train["quaternions"], train["xs"]
            Rte, Ste, Xte = test["bloch_direct"], test["states"], test["xs"]

            common_train_weights = state_motion(Str) ** 2
            common_test_weights = state_motion(Ste) ** 2

            results = {
                "z": None,
                "best_pauli": best_pauli_axis(Rtr, weights=common_train_weights, xs=Xtr),
                "bloch_pca": unweighted_bloch_pca_axis(Rtr, xs=Xtr),
                "state_weighted": state_motion_weighted_axis(Rtr, Str, xs=Xtr),
                "raw_quaternion": raw_quaternion_weighted_axis(Rtr, Qtr, xs=Xtr),
                "quotient_quaternion": quotient_quaternion_weighted_axis(Rtr, Str, xs=Xtr),
            }
            axes = {"z": np.array([0.0, 0.0, 1.0])}
            for method, result in results.items():
                if method == "z":
                    continue
                axes[method] = result.axis if result.identifiable else None

            for method, axis in axes.items():
                result = results[method]
                identifiable = axis is not None
                test_energy = (
                    weighted_tangent_energy(Rte, axis, weights=common_test_weights, xs=Xte)
                    if identifiable else np.nan
                )
                test_hidden = (
                    observable_hidden_motion_ratio(Rte, Ste, axis)
                    if identifiable else np.nan
                )
                rows.append({
                    "depth": int(depth),
                    "seed": int(seed),
                    "method": method,
                    "identifiable": bool(identifiable),
                    "train_eigengap": float(result.eigengap) if result is not None else np.nan,
                    "train_leading_energy": float(result.energy) if result is not None else np.nan,
                    "test_common_energy": float(test_energy),
                    "test_observable_hidden_ratio": float(test_hidden),
                })

            pca_axis = axes["bloch_pca"]
            quotient_axis = axes["quotient_quaternion"]
            state_axis = axes["state_weighted"]
            if pca_axis is not None and quotient_axis is not None:
                angle_q_pca = axis_angle(quotient_axis, pca_axis)
            else:
                angle_q_pca = np.nan
            if state_axis is not None and quotient_axis is not None:
                angle_q_state = axis_angle(quotient_axis, state_axis)
            else:
                angle_q_state = np.nan
            for row in rows[-6:]:
                row["angle_quotient_to_pca_rad"] = float(angle_q_pca)
                row["angle_quotient_to_state_rad"] = float(angle_q_state)

    with (TAB / "05_multiseed_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[int, int], dict[str, dict[str, float | int | str | bool]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["depth"]), int(row["seed"]))][str(row["method"])] = row

    paired_rows = []
    for (depth, seed), group in grouped.items():
        q = group["quotient_quaternion"]
        p = group["bloch_pca"]
        s = group["state_weighted"]
        paired_rows.append({
            "depth": depth,
            "seed": seed,
            "quotient_minus_pca_test_energy": float(q["test_common_energy"] - p["test_common_energy"]),
            "quotient_minus_state_test_energy": float(q["test_common_energy"] - s["test_common_energy"]),
            "angle_quotient_to_pca_rad": float(q["angle_quotient_to_pca_rad"]),
            "angle_quotient_to_state_rad": float(q["angle_quotient_to_state_rad"]),
            "quotient_identifiable": bool(q["identifiable"]),
            "pca_identifiable": bool(p["identifiable"]),
        })

    summary = []
    for depth in depths:
        rr = [r for r in paired_rows if r["depth"] == depth]
        diffs = np.asarray([r["quotient_minus_pca_test_energy"] for r in rr], dtype=float)
        ci_low, ci_high = _paired_bootstrap_ci(diffs, seed=2026 + int(depth))
        summary.append({
            "depth": int(depth),
            "n_pairs": len(rr),
            "mean_quotient_minus_pca_test_energy": float(np.mean(diffs)),
            "median_quotient_minus_pca_test_energy": float(np.median(diffs)),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "fraction_quotient_gt_pca": float(np.mean(diffs > 1e-12)),
            "median_angle_quotient_to_pca_rad": float(np.nanmedian([r["angle_quotient_to_pca_rad"] for r in rr])),
            "max_angle_quotient_to_state_rad": float(np.nanmax([r["angle_quotient_to_state_rad"] for r in rr])),
            "fraction_quotient_identifiable": float(np.mean([r["quotient_identifiable"] for r in rr])),
            "fraction_pca_identifiable": float(np.mean([r["pca_identifiable"] for r in rr])),
        })

    with (TAB / "05_multiseed_statistics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes_plot = plt.subplots(1, 2, figsize=(10, 4))
    diff_data = [
        [r["quotient_minus_pca_test_energy"] for r in paired_rows if r["depth"] == d]
        for d in depths
    ]
    axes_plot[0].boxplot(diff_data, tick_labels=[str(d) for d in depths], showfliers=False)
    axes_plot[0].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes_plot[0].set_xlabel("QRU depth")
    axes_plot[0].set_ylabel(r"$E_{test}(v_{quot})-E_{test}(v_{PCA})$")

    angle_data = [
        np.degrees([r["angle_quotient_to_pca_rad"] for r in paired_rows if r["depth"] == d])
        for d in depths
    ]
    axes_plot[1].boxplot(angle_data, tick_labels=[str(d) for d in depths], showfliers=False)
    axes_plot[1].set_xlabel("QRU depth")
    axes_plot[1].set_ylabel("Axis angle (degrees)")
    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    fig.savefig(FIG / "05_multiseed_energy_gain.pdf")
    plt.close(fig)

    print("Paired out-of-sample quotient-vs-PCA summary")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
