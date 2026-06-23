# QRU geometry-aware coherent registerization

Paper-focused repo for **Geometry-Aware Coherent Registerization of Single-Qubit Data Re-Uploading Models**.

Claim scope: gauge-aware signed Bloch readout + finite coherent signed-register interface + verified small coherent demonstrator. No quantum advantage, no quaternion-performance advantage, no scalable QAE-decoder claim.

## Structure

```text
paper/                         LaTeX source and curated paper figures
src/qru_registerization/        geometry, QRU and registerization code
experiments/                    scripts regenerating paper figures/tables
outputs/                        generated figures, tables and manifest
tests/                          regression tests
```

## Install

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Reproduce

```bash
python -m pytest -q
python experiments/run_all.py
latexmk -pdf -cd paper/main.tex
```

If external pytest plugins hang in a managed environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Active experiments

```text
01_gauge_validation.py          gauge-only SU(2) motion and quotient correction
02_readout_selection.py         projection blindness and tangent-axis selection
03_coherent_registerization.py  coherent QAE, reversible decoder, error/resources
04_coherent_composition.py      register-controlled evolution and coherence echo
05_readout_generalization.py    train/test random-QRU readout comparison
06_bloch_register_diagnostic.py geometry-to-register diagnostic instance
```

`03_coherent_registerization.py` writes regenerated full-resource values, block-level resource tables, and downstream-bound grids. `_plot_03_outputs.py` renders the two 03 figures from the persisted CSV tables.
