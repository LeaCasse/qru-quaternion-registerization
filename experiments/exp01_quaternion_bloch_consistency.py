from __future__ import annotations

import json
import numpy as np

from _common import DAT, TAB
from qru_registerization.gates import random_qru_params
from qru_registerization.pipeline import compute_paths


def main() -> None:
    xs = np.linspace(-np.pi, np.pi, 201)
    params = random_qru_params(depth=4, seed=11, scale=0.9)
    paths = compute_paths(params, xs)
    err = np.linalg.norm(paths["bloch_direct"] - paths["bloch_quaternion"], axis=1)
    summary = {
        "depth": 4,
        "n_points": int(len(xs)),
        "max_bloch_consistency_error": float(err.max()),
        "mean_bloch_consistency_error": float(err.mean()),
    }
    (DAT / "exp01_params.npy").write_bytes(b"")
    np.save(DAT / "exp01_params.npy", params)
    (TAB / "exp01_quaternion_bloch_consistency.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
