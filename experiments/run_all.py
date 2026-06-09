from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_gauge_validation.py",
    "02_readout_selection.py",
    "05_readout_generalization.py",
    "03_coherent_registerization.py",
    "04_coherent_composition.py",
]

for script in SCRIPTS:
    print(f"Running {script}...", flush=True)
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / script))}"
    status = os.system(command)
    if status != 0:
        raise SystemExit(f"{script} failed with status {status}")

print("All experiments completed.")
