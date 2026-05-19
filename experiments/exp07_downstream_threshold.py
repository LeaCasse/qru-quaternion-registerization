from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.axis_selection import variance_axis
from qru_registerization.io import write_csv


def main() -> None:
    xs = np.linspace(-np.pi, np.pi, 301)
    params = random_qru_params(depth=4, seed=41, scale=1.0)
    paths = compute_paths(params, xs)
    r = paths["bloch_direct"]
    v_var = variance_axis(r)
    v_z = np.array([0.0, 0.0, 1.0])
    reg_var = registerize_path(paths["states"], r, v_var, m=5)
    reg_z = registerize_path(paths["states"], r, v_z, m=5)
    tau = 0.0
    exact_flag = (reg_var["s"] > tau).astype(int)
    quant_flag = (reg_var["s_quant"] > tau).astype(int)
    z_flag = (reg_z["s"] > tau).astype(int)
    rows = [
        {"method": "v_var_quantized_m5", "disagreement_rate_vs_exact_v_var": float(np.mean(quant_flag != exact_flag))},
        {"method": "z_axis_exact", "disagreement_rate_vs_exact_v_var": float(np.mean(z_flag != exact_flag))},
    ]
    write_csv(TAB / "exp07_downstream_threshold.csv", rows)

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, reg_var["s"], label=r"$s_{v_{var}}(x)$")
    plt.step(xs, reg_var["s_quant"], where="mid", label=r"quantized $s_{v_{var}}$, m=5")
    plt.plot(xs, reg_z["s"], label=r"$s_z(x)$", alpha=0.8)
    plt.axhline(tau, linestyle="--", linewidth=1.0, label="threshold")
    plt.xlabel("x")
    plt.ylabel("signed coordinate")
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig07_downstream_threshold.pdf")
    plt.close()


if __name__ == "__main__":
    main()
