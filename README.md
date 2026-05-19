# QRU quaternion-diagnostic signed registerization

Operational code for the paper:

**Quaternion-Diagnostic Signed Registerization of Single-Qubit Data Re-Uploading Models**

Core pipeline:

```text
QRU U_theta(x)
  -> quaternion path q_theta(x) in S^3
  -> hidden-motion diagnostics delta_q
  -> quaternion-guided readout axis v_quat
  -> signed coordinate s_v(x)
  -> QAE-compatible probability p_v(x)=(1+s_v(x))/2
  -> signed fixed-point register estimate
  -> downstream QAOA toy layer
```

## Install

```bash
pip install -e .
pip install -r requirements.txt
```

## Validate

```bash
python -m pytest -q
```

## Reproduce experiments

```bash
python experiments/run_all.py
```

Generated files are written to:

```text
outputs/figures/
outputs/tables/
outputs/data/
```

## Repository structure

```text
src/qru_registerization/     # reusable research code
experiments/                 # paper-aligned experiments
notebooks/                   # executable pedagogical walkthrough
outputs/                     # generated artifacts
```

## Main scripts

```text
01_hidden_motion_diagnostics.py   # q-motion vs visible readout motion
02_readout_axis_comparison.py     # e_x,e_y,e_z vs v_quat
03_registerization_precision.py   # signed fixed-point error
04_qru_qaoa_case_study.py         # QRU -> QAOA downstream comparison
05_multiseed_statistics.py        # robustness over seeds/depths
```
