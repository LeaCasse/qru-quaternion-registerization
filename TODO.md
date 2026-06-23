# TODO — current pass status

## Done in pass 2

- Qiskit gate validated: `qiskit==1.4.6` pinned.
- Added reusable builders:
  - `build_phase_to_signed_decoder_circuit`
  - `build_signed_downstream_block_circuit`
- Added block-level resource outputs:
  - `outputs/tables/coherent_resource_breakdown.csv`
  - `outputs/tables/coherent_resource_breakdown.tex`
- Added downstream-bound grid:
  - `outputs/tables/downstream_bound_grid.csv`
- Added tests:
  - `tests/test_downstream_bound_grid.py`
  - `tests/test_uncomputation_grid.py`
- `03_precision_resource_tradeoff.csv` now contains regenerated validation values for all three precision pairs, not NaN placeholders.
- `paper/main.tex` compiles in this environment.
- `outputs/metadata/run_manifest.json` records dependencies, Qiskit version, transpile config and expected outputs.

## Validated commands in this pass

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiments/03_coherent_registerization.py
PYTHONPATH=src python experiments/_plot_03_outputs.py
PYTHONPATH=src python experiments/04_coherent_composition.py
latexmk -pdf -cd paper/main.tex
```

## Key numerical state

- Full pipeline resources:
  - `(3,2)`: 8 qubits, depth 795, CX 326
  - `(4,3)`: 10 qubits, depth 6156, CX 2170
  - `(5,4)`: 12 qubits, depth 19443, CX 6513
- Lookup decoder dominates CX:
  - `(3,2)`: 150 CX per lookup
  - `(4,3)`: 1050 CX per lookup
  - `(5,4)`: 3210 CX per lookup
- Coherent validation remains near machine precision:
  - signed-register reset probability > `1 - 2e-13`
  - state fidelity > `1 - 2e-13`
- Probability robustness unchanged:
  - mean signed error on uniform grid: `0.244`, `0.139`, `0.0829`
  - fraction with error <= `0.02`: `1.5%`, `4.5%`, `11.4%`

## Next P0

1. Insert or reference `coherent_resource_breakdown.tex` in `paper/main.tex`, replacing hardcoded resource prose if useful.
2. Decide whether Fig. `03_registerization_precision.pdf` is paper/appendix only or stays output-only.
3. On Windows, fix LaTeX toolchain by installing real system Perl for MiKTeX `latexmk`; `pip install perl` is not sufficient.
4. Re-run locally:

```bat
python -m pytest -q
python experiments\run_all.py
latexmk -pdf -cd paper\main.tex
```

## P1 only after P0

- Optional trained-QRU state-trajectory experiment.
- Use trained QRU only if the paper keeps any wording about readout performance on learned tasks.

## Still excluded

- No quaternion-performance claim.
- No raw quaternion loss.
- No Rall/QSP/QSVT implementation in `main`.
- No QAOA case study in this paper.
- No Quantastica validation or resource claims.
