from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB, DAT
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths
from qru_registerization.quaternion_geometry import quaternion_path_distances
from qru_registerization.io import write_csv


def pca3(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def main() -> None:
    xs = np.linspace(-np.pi, np.pi, 301)
    params = random_qru_params(depth=5, seed=5, scale=0.8)
    paths = compute_paths(params, xs)
    q = paths["quaternions"]
    r = paths["bloch_direct"]
    qp = pca3(q)
    d = quaternion_path_distances(q)

    np.save(DAT / "exp03_params.npy", params)
    write_csv(
        TAB / "exp03_quaternion_path_distances.csv",
        [{"segment": int(i), "distance_s3": float(di)} for i, di in enumerate(d)],
    )

    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(qp[:, 0], qp[:, 1], qp[:, 2], linewidth=1.5)
    ax.scatter(qp[0, 0], qp[0, 1], qp[0, 2], s=25, label="start")
    ax.scatter(qp[-1, 0], qp[-1, 1], qp[-1, 2], s=25, label="end")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig03a_quaternion_path_pca.pdf")
    plt.close()

    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=1.5)
    ax.scatter(r[0, 0], r[0, 1], r[0, 2], s=25, label="start")
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], s=25, label="end")
    ax.set_xlabel(r"$\langle X\rangle$")
    ax.set_ylabel(r"$\langle Y\rangle$")
    ax.set_zlabel(r"$\langle Z\rangle$")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig03b_bloch_path.pdf")
    plt.close()


if __name__ == "__main__":
    main()
