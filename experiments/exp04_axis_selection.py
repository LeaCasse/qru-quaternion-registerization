from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB, DAT
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.axis_selection import variance_axis, tangent_axis, projected_variance, projected_tangent_energy
from qru_registerization.io import write_csv


def main() -> None:
    xs = np.linspace(-np.pi, np.pi, 301)
    params = random_qru_params(depth=5, seed=19, scale=1.0)
    paths = compute_paths(params, xs)
    r = paths["bloch_direct"]
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "v_var": variance_axis(r),
        "v_tan": tangent_axis(r, xs),
    }
    rows = []
    plt.figure(figsize=(6.8, 4.3))
    for name, v in axes.items():
        s = r @ v
        rows.append(
            {
                "axis": name,
                "vx": float(v[0]),
                "vy": float(v[1]),
                "vz": float(v[2]),
                "projected_variance": projected_variance(r, v),
                "projected_tangent_energy": projected_tangent_energy(r, v, xs),
                "sign_changes": int(np.sum(np.signbit(s[1:]) != np.signbit(s[:-1]))),
            }
        )
        plt.plot(xs, s, label=name)
    write_csv(TAB / "exp04_axis_selection_metrics.csv", rows)
    np.save(DAT / "exp04_params.npy", params)
    plt.xlabel("x")
    plt.ylabel(r"$s_v(x)=v\cdot r_\theta(x)$")
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(FIG / "fig04_axis_selection.pdf")
    plt.close()


if __name__ == "__main__":
    main()
