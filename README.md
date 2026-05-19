# qru-quaternion-registerization

Operational code for quaternion-guided signed registerization of single-qubit QRU outputs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Generate all outputs

```bash
python experiments/run_all.py
```

Outputs are written to:

```text
outputs/figures/
outputs/tables/
outputs/data/
```
