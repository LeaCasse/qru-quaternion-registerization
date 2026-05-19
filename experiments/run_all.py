from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_hidden_motion_diagnostics.py",
    "02_readout_axis_comparison.py",
    "03_registerization_precision.py",
    "04_qru_qaoa_case_study.py",
    "05_multiseed_statistics.py",
]

for script in SCRIPTS:
    print(f"Running {script}...")
    runpy.run_path(str(ROOT / script), run_name="__main__")
print("All experiments completed.")
