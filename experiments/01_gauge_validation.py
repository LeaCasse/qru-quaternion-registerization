from __future__ import annotations

import csv
import matplotlib.pyplot as plt
import numpy as np

from _common import FIG, TAB
from qru_registerization.bloch import bloch_vector
from qru_registerization.gates import ry, rz
from qru_registerization.quaternion_geometry import (
    bloch_geodesic_distance,
    fubini_study_distance,
    quotient_quaternion_distance,
    relative_rotation_angle,
    su2_to_quaternion,
)


def main(n_x: int = 161) -> None:
    """Gauge-only control: U(x)=V Rz(x) changes in SU(2) but not as a state."""
    xs = np.linspace(-np.pi, np.pi, n_x)
    V = ry(0.7)
    ket0 = np.array([1.0, 0.0], dtype=complex)

    unitaries = np.asarray([V @ rz(float(x)) for x in xs])
    quaternions = np.asarray([su2_to_quaternion(U) for U in unitaries])
    states = np.asarray([U @ ket0 for U in unitaries])
    bloch = np.asarray([bloch_vector(psi) for psi in states])

    raw = np.asarray([
        relative_rotation_angle(quaternions[i], quaternions[i + 1])
        for i in range(n_x - 1)
    ])
    quotient = np.asarray([
        quotient_quaternion_distance(unitaries[i], unitaries[i + 1])
        for i in range(n_x - 1)
    ])
    fubini_study = np.asarray([
        fubini_study_distance(states[i], states[i + 1])
        for i in range(n_x - 1)
    ])
    bloch_distance = np.asarray([
        bloch_geodesic_distance(bloch[i], bloch[i + 1])
        for i in range(n_x - 1)
    ])
    x_mid = 0.5 * (xs[:-1] + xs[1:])

    assert float(np.max(quotient)) < 1e-10
    assert float(np.max(fubini_study)) < 1e-10
    assert float(np.max(bloch_distance)) < 1e-7
    assert float(np.sum(raw)) > 6.0

    rows = [
        {
            "x_mid": float(x_mid[i]),
            "raw_rotation_angle": float(raw[i]),
            "quotient_distance": float(quotient[i]),
            "fubini_study_distance": float(fubini_study[i]),
            "bloch_distance": float(bloch_distance[i]),
        }
        for i in range(n_x - 1)
    ]
    with (TAB / "01_gauge_only_motion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = [{
        "total_raw_rotation": float(np.sum(raw)),
        "max_quotient_distance": float(np.max(quotient)),
        "max_fubini_study_distance": float(np.max(fubini_study)),
        "max_bloch_distance": float(np.max(bloch_distance)),
    }]
    with (TAB / "01_gauge_only_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x_mid, raw, label=r"raw relative rotation $\delta_{\mathrm{rot}}$")
    ax.plot(x_mid, quotient, label=r"quotient distance $\delta_{\mathrm{quot}}$")
    ax.plot(x_mid, fubini_study, linestyle="--", label=r"Fubini--Study $\delta_{\mathrm{FS}}$")
    ax.plot(x_mid, bloch_distance, linestyle=":", label=r"Bloch geodesic $\delta_{\mathrm{Bloch}}$")
    ax.set_xlabel("x")
    ax.set_ylabel("consecutive-point distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_gauge_only_motion.pdf")
    fig.savefig(FIG / "01_gauge_only_motion.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
