# QRU geometry-aware coherent registerization

Code accompanying the paper **Geometry-Aware Coherent Registerization of Single-Qubit Data Re-Uploading Models**.

The repository implements two components:

1. gauge-aware selection of a signed QRU readout;
2. coherent conversion of that readout into a finite register controlling a downstream quantum operation.

## Install

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Validate

```bash
python -m pytest -q
```

## Reproduce all experiments

```bash
python experiments/run_all.py
```

The aggregate runner may take several minutes because it launches Qiskit statevector and transpilation checks. To recompute the full expensive resource sweep instead of using the validated reference counts, run:

```bash
python experiments/03_coherent_registerization.py --full-resources
```

Generated results are written to `outputs/figures/` and `outputs/tables/`.

## Experiments

```text
01_gauge_validation.py          gauge-only motion and quotient correction
02_readout_selection.py         controlled projection blindness and axis selection
03_coherent_registerization.py  QAE, reversible signed decoding, error and robustness
04_coherent_composition.py      register-controlled evolution and coherence test
05_readout_generalization.py    train/test multi-seed readout comparison
```

## Paper figures

The four PDFs in `figures/` are the curated figures referenced by `main.tex`. They are regenerated from the experiment outputs; no notebook is required.
