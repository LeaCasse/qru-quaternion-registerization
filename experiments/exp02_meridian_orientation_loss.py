from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB
from qru_registerization.io import write_csv


def main() -> None:
    phi = np.linspace(-np.pi, np.pi, 401)
    x_coord = np.sin(phi)
    z_coord = np.cos(phi)
    rows = [{"phi": float(a), "x_coordinate": float(b), "z_coordinate": float(c)} for a, b, c in zip(phi, x_coord, z_coord)]
    write_csv(TAB / "exp02_meridian_orientation_loss.csv", rows)

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(phi, z_coord, label=r"$\langle Z\rangle=\cos\phi$")
    plt.plot(phi, x_coord, label=r"$\langle X\rangle=\sin\phi$")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel(r"$\phi$")
    plt.ylabel("coordinate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig02_meridian_orientation_loss.pdf")
    plt.close()


if __name__ == "__main__":
    main()
