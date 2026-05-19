from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.quaternion_diagnostics import quaternion_guided_axis
from qru_registerization.fixed_point import max_quantization_error_bound


def main(seed: int = 13, depth: int = 5, n_x: int = 161) -> None:
    xs = np.linspace(-np.pi, np.pi, n_x)
    params = random_qru_params(depth=depth, seed=seed, scale=0.9)
    paths = compute_paths(params, xs)
    R = paths["bloch_direct"]
    Q = paths["quaternions"]
    states = paths["states"]
    vq = quaternion_guided_axis(R, Q, xs)

    rows = []
    for m in range(2, 13):
        reg = registerize_path(states, R, vq, m=m)
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
        writer.writeheader(); writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    m = np.array([r["m_bits"] for r in rows])
    ax.semilogy(m, [r["max_abs_error"] for r in rows], marker="o", label="max error")
    ax.semilogy(m, [r["bound_2_minus_m"] for r in rows], linestyle="--", label=r"$2^{-m}$ bound")
    ax.set_xlabel("magnitude bits m")
    ax.set_ylabel("absolute error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_registerization_precision.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
