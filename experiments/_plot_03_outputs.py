from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)


def _rows(name: str) -> list[dict[str, str]]:
    with (TAB / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_probability_robustness() -> None:
    robustness_rows = _rows("03_probability_robustness.csv")
    summary_rows = _rows("03_probability_robustness_summary.csv")
    configs = ((3, 2), (4, 3), (5, 4))

    fig2, axes2 = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for phase_bits, magnitude_bits in configs:
        selected = [
            r for r in robustness_rows
            if r["source"] == "analytic_grid"
            and int(r["phase_bits"]) == phase_bits
            and int(r["magnitude_bits"]) == magnitude_bits
        ]
        label = rf"$({phase_bits},{magnitude_bits})$"
        axes2[0, 0].plot(
            [_float(r, "exact_probability") for r in selected],
            [_float(r, "expected_total_signed_abs_error") for r in selected],
            label=label,
        )
        axes2[0, 1].plot(
            [_float(r, "exact_probability") for r in selected],
            [max(_float(r, "expected_downstream_state_infidelity"), np.finfo(float).eps) for r in selected],
            label=label,
        )
    axes2[0, 0].set_xlabel("exact probability $p$")
    axes2[0, 0].set_ylabel("expected signed absolute error")
    axes2[0, 0].set_title("Error across the full probability interval")
    axes2[0, 0].legend(fontsize=8)
    axes2[0, 1].set_xlabel("exact probability $p$")
    axes2[0, 1].set_ylabel("expected downstream infidelity")
    axes2[0, 1].set_title("Downstream impact across $p$")
    axes2[0, 1].legend(fontsize=8)

    positions = np.arange(len(configs))
    labels_cfg = [f"({a},{b})" for a, b in configs]
    grid_summary = [r for r in summary_rows if r["source"] == "analytic_grid"]
    qru_summary = [r for r in summary_rows if r["source"] == "qru"]
    width = 0.36
    axes2[1, 0].bar(positions - width / 2, [_float(r, "mean_total_signed_abs_error") for r in grid_summary], width, label="uniform p grid")
    axes2[1, 0].bar(positions + width / 2, [_float(r, "mean_total_signed_abs_error") for r in qru_summary], width, label="QRU samples")
    axes2[1, 0].set_xticks(positions, labels_cfg)
    axes2[1, 0].set_xlabel(r"$(m_\phi,m_s)$")
    axes2[1, 0].set_ylabel("mean signed absolute error")
    axes2[1, 0].set_title("Uniform-grid and QRU error")
    axes2[1, 0].legend(fontsize=8)

    axes2[1, 1].bar(positions - width / 2, [_float(r, "fraction_total_error_le_0_02") for r in grid_summary], width, label="uniform p grid")
    axes2[1, 1].bar(positions + width / 2, [_float(r, "fraction_total_error_le_0_02") for r in qru_summary], width, label="QRU samples")
    axes2[1, 1].set_xticks(positions, labels_cfg)
    axes2[1, 1].set_ylim(0.0, 1.0)
    axes2[1, 1].set_xlabel(r"$(m_\phi,m_s)$")
    axes2[1, 1].set_ylabel(r"fraction with $\epsilon_s\leq0.02$")
    axes2[1, 1].set_title("Coverage of a 0.02 signed-error target")
    axes2[1, 1].legend(fontsize=8)
    fig2.subplots_adjust(hspace=0.42, wspace=0.32)
    fig2.savefig(FIG / "03_probability_robustness.pdf")
    plt.close(fig2)


def plot_registerization_precision() -> None:
    rows = _rows("03_registerization_precision.csv")
    validation_rows = _rows("03_basis_rotation_validation.csv")
    qae_rows = _rows("03_coherent_qae_validation.csv")
    tradeoff_rows = _rows("03_precision_resource_tradeoff.csv")

    phase_bits_values = sorted({int(r["phase_bits"]) for r in qae_rows})
    qae_summary = {}
    for m in phase_bits_values:
        selected = [r for r in qae_rows if int(r["phase_bits"]) == m]
        qae_summary[m] = {
            "exact_probability": _float(selected[0], "exact_amplitude"),
            "absolute_mean_error": _float(selected[0], "absolute_mean_error"),
            "distribution": {},
            "amplitude_estimates": {},
        }
        for r in selected:
            idx = int(r["phase_index"])
            qae_summary[m]["distribution"][idx] = _float(r, "phase_probability")
            qae_summary[m]["amplitude_estimates"][idx] = _float(r, "decoded_amplitude")

    fig, axes = plt.subplots(3, 2, figsize=(11.4, 11.2))
    axes = axes.ravel()
    m_vals = np.array([int(r["m_bits"]) for r in rows])
    axes[0].semilogy(m_vals, [_float(r, "max_abs_error") for r in rows], marker="o", label="max error")
    axes[0].semilogy(m_vals, [_float(r, "bound_2_minus_m") for r in rows], linestyle="--", label=r"$2^{-m}$ bound")
    axes[0].set_xlabel("magnitude bits $m$")
    axes[0].set_ylabel("absolute error")
    axes[0].set_title("Signed fixed-point quantization")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(
        [int(r["sample"]) for r in validation_rows],
        [max(_float(r, "observable_operator_error"), np.finfo(float).eps) for r in validation_rows],
        label=r"$\|B_v^\dagger ZB_v-v\cdot\sigma\|_2$",
    )
    axes[1].semilogy(
        [int(r["sample"]) for r in validation_rows],
        [max(_float(r, "probability_error"), np.finfo(float).eps) for r in validation_rows],
        label="probability mismatch",
    )
    axes[1].set_xlabel("random validation case")
    axes[1].set_ylabel("numerical residual")
    axes[1].set_title("Directional basis rotation")
    axes[1].legend(fontsize=8)

    axes[2].semilogy(
        phase_bits_values,
        [qae_summary[m]["absolute_mean_error"] for m in phase_bits_values],
        marker="o",
    )
    axes[2].set_xlabel("QAE phase bits")
    axes[2].set_ylabel(r"$|\mathbb{E}[\hat p]-p|$")
    axes[2].set_title(rf"Coherent QAE precision, exact $p={qae_summary[2]['exact_probability']:.3f}$")

    for phase_bits, marker in ((2, "o"), (4, "s")):
        summary = qae_summary[phase_bits]
        aggregated = {}
        for index, weight in summary["distribution"].items():
            estimate = round(summary["amplitude_estimates"][index], 14)
            aggregated[estimate] = aggregated.get(estimate, 0.0) + weight
        estimates = sorted(aggregated)
        axes[3].plot(estimates, [aggregated[value] for value in estimates], marker=marker, label=f"phase bits={phase_bits}")
    axes[3].axvline(qae_summary[2]["exact_probability"], linestyle="--", label="exact probability")
    axes[3].set_xlabel(r"decoded amplitude $\hat p$")
    axes[3].set_ylabel("probability mass")
    axes[3].set_title("Decoded coherent QAE distribution")
    axes[3].legend(fontsize=8)

    labels = [f"({int(r['phase_bits'])},{int(r['magnitude_bits'])})" for r in tradeoff_rows]
    positions = np.arange(len(tradeoff_rows))
    axes[4].plot(positions, [_float(r, "compiled_cx_count") for r in tradeoff_rows], marker="o", label="CX count")
    axes[4].plot(positions, [_float(r, "compiled_depth") for r in tradeoff_rows], marker="s", label="compiled depth")
    axes[4].set_xticks(positions, labels)
    axes[4].set_xlabel(r"$(m_\phi,m_s)$")
    axes[4].set_ylabel("compiled resources")
    axes[4].set_title("Truth-table decoder resource growth")
    axes[4].legend(fontsize=8)

    axes[5].semilogy(positions, [_float(r, "expected_qae_signed_abs_error") for r in tradeoff_rows], marker="o", label="QAE contribution")
    axes[5].semilogy(positions, [_float(r, "expected_decoder_abs_error") for r in tradeoff_rows], marker="s", label="decoder contribution")
    axes[5].semilogy(positions, [_float(r, "expected_total_signed_abs_error") for r in tradeoff_rows], marker="^", label="total signed error")
    axes[5].set_xticks(positions, labels)
    axes[5].set_xlabel(r"$(m_\phi,m_s)$")
    axes[5].set_ylabel("expected absolute error")
    axes[5].set_title("Error decomposition")
    axes[5].legend(fontsize=7)

    fig.subplots_adjust(hspace=0.48, wspace=0.34)
    fig.savefig(FIG / "03_registerization_precision.pdf")
    plt.close(fig)


def main() -> None:
    plot_probability_robustness()
    plot_registerization_precision()


if __name__ == "__main__":
    main()
