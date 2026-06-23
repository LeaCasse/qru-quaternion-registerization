from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SRC = REPO / "src"
META = REPO / "outputs" / "metadata"
META.mkdir(parents=True, exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCRIPTS = [
    ("05_readout_generalization.py", False, []),
    ("01_gauge_validation.py", False, []),
    ("02_readout_selection.py", False, []),
    ("06_bloch_register_diagnostic.py", False, []),
    ("03_coherent_registerization.py", True, []),
    ("_plot_03_outputs.py", False, []),
    ("04_coherent_composition.py", True, []),
]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _run_script(script: str, args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=REPO, env=env, check=True)


qiskit_available = importlib.util.find_spec("qiskit") is not None
runs = []
for script, needs_qiskit, args in SCRIPTS:
    if needs_qiskit and not qiskit_available:
        print(f"Skipping {script}: qiskit is not installed.", flush=True)
        runs.append({"script": script, "status": "skipped", "reason": "qiskit_missing"})
        continue
    print(f"Running {script}...", flush=True)
    _run_script(script, args)
    runs.append({"script": script, "status": "passed", "args": args})

from qru_registerization.transpile_config import (  # noqa: E402
    PAPER_BASIS_GATES,
    PAPER_OPTIMIZATION_LEVEL,
    PAPER_SEED_TRANSPILER,
)

expected_outputs = [
    "outputs/figures/01_gauge_only_motion.pdf",
    "outputs/figures/02_projection_blindness.pdf",
    "outputs/figures/03_probability_robustness.pdf",
    "outputs/figures/03_registerization_precision.pdf",
    "outputs/figures/04_register_controlled_z_validation.pdf",
    "outputs/figures/05_multiseed_energy_gain.pdf",
    "outputs/figures/06_bloch_register_diagnostic.pdf",
    "outputs/tables/03_precision_resource_tradeoff.csv",
    "outputs/tables/coherent_resource_breakdown.csv",
    "outputs/tables/coherent_resource_breakdown.tex",
    "outputs/tables/downstream_bound_grid.csv",
    "outputs/tables/04_two_branch_coherence_summary.csv",
]

manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "git_hash": _git_hash(),
    "dependencies": {
        "numpy": _version("numpy"),
        "matplotlib": _version("matplotlib"),
        "qiskit": _version("qiskit"),
        "pytest": _version("pytest"),
    },
    "qiskit_available": qiskit_available,
    "transpile_config": {
        "basis_gates": PAPER_BASIS_GATES,
        "optimization_level": PAPER_OPTIMIZATION_LEVEL,
        "seed_transpiler": PAPER_SEED_TRANSPILER,
    },
    "scripts": runs,
    "expected_outputs": {path: (REPO / path).exists() for path in expected_outputs},
}
(META / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("All available experiments completed.", flush=True)
print(f"Manifest: {META / 'run_manifest.json'}", flush=True)
os._exit(0)
