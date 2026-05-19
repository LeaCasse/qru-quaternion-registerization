from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import FIG, TAB
from qru_registerization.fixed_point import quantize_signed_magnitude, max_quantization_error_bound
from qru_registerization.io import write_csv


def main() -> None:
    s = np.linspace(-1.0, 1.0, 2001)
    rows = []
    for m in range(2, 13):
        q = np.array([quantize_signed_magnitude(si, m) for si in s])
        err = np.abs(q - s)
        rows.append(
            {
                "bits": m,
                "max_abs_error": float(err.max()),
                "mean_abs_error": float(err.mean()),
                "bound": max_quantization_error_bound(m),
            }
        )
    write_csv(TAB / "exp05_register_precision.csv", rows)

    bits = np.array([r["bits"] for r in rows])
    maxerr = np.array([r["max_abs_error"] for r in rows])
    bound = np.array([r["bound"] for r in rows])
    plt.figure(figsize=(6.2, 4.0))
    plt.semilogy(bits, maxerr, marker="o", label="observed max error")
    plt.semilogy(bits, bound, marker="s", label=r"$2^{-m}$ bound")
    plt.xlabel("fractional bits m")
    plt.ylabel("absolute error")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig05_register_precision.pdf")
    plt.close()


if __name__ == "__main__":
    main()
