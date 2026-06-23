from __future__ import annotations

PAPER_BASIS_GATES = ["rz", "sx", "x", "cx"]
PAPER_OPTIMIZATION_LEVEL = 0
PAPER_SEED_TRANSPIlER = 17  # deprecated spelling kept below for old notebooks if any
PAPER_SEED_TRANSPILER = PAPER_SEED_TRANSPIlER

PAPER_TRANSPILE_KWARGS = {
    "basis_gates": PAPER_BASIS_GATES,
    "optimization_level": PAPER_OPTIMIZATION_LEVEL,
    "seed_transpiler": PAPER_SEED_TRANSPILER,
}
