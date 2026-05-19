from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = [
    "exp01_quaternion_bloch_consistency.py",
    "exp02_meridian_orientation_loss.py",
    "exp03_qru_quaternion_path.py",
    "exp04_axis_selection.py",
    "exp05_register_precision.py",
    "exp06_readout_comparison.py",
    "exp07_downstream_threshold.py",
]


def main() -> None:
    for exp in EXPS:
        print(f"[run] {exp}")
        subprocess.run([sys.executable, str(ROOT / "experiments" / exp)], check=True)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
