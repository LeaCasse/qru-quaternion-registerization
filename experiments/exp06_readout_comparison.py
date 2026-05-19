from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths, registerize_path
from qru_registerization.axis_selection import variance_axis
from qru_registerization.amplitude_interface import ideal_ae_error_model, signed_coordinate_from_probability, probability_from_signed_coordinate
from qru_registerization.io import write_csv


def shot_estimate_s(s: np.ndarray, shots: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = (1 + s) / 2
    counts = rng.binomial(shots, p)
    return 2 * counts / shots - 1


def main() -> None:
    xs = np.linspace(-np.pi, np.pi, 201)
    params = random_qru_params(depth=4, seed=23, scale=0.8)
    paths = compute_paths(params, xs)
    r = paths["bloch_direct"]
    v = variance_axis(r)
    reg = registerize_path(paths["states"], r, v, m=6)
    s = reg["s"]
    s_shot = shot_estimate_s(s, shots=256, seed=31)
    p = np.array([probability_from_signed_coordinate(si) for si in s])
    p_ae = np.array([ideal_ae_error_model(pi, epsilon=1/64, seed=i) for i, pi in enumerate(p)])
    s_ae = np.array([signed_coordinate_from_probability(pi) for pi in p_ae])
    rows = []
    for name, arr in [("shot_256", s_shot), ("ideal_ae_eps_1_64", s_ae), ("fixed_point_m6", reg["s_quant"] )]:
        err = np.abs(arr - s)
        rows.append({"method": name, "mean_abs_error": float(err.mean()), "max_abs_error": float(err.max())})
    write_csv(TAB / "exp06_readout_comparison.csv", rows)

    plt.figure(figsize=(7.0, 4.3))
    plt.plot(xs, s, label="exact signed coordinate", linewidth=2)
    plt.plot(xs, s_shot, label="shot estimate, N=256", alpha=0.8)
    plt.plot(xs, s_ae, label="ideal AE-style model, ε=1/64", alpha=0.8)
    plt.step(xs, reg["s_quant"], where="mid", label="fixed point, m=6", alpha=0.8)
    plt.xlabel("x")
    plt.ylabel(r"$s_v(x)$")
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig06_readout_comparison.pdf")
    plt.close()


if __name__ == "__main__":
    main()
