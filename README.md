# Simulation Assets — Two Roads to Quantum Safety

This directory contains the simulation scripts used to produce Tables II–VI and
Figures 3–7 of the paper "Two Roads to Quantum Safety: Where QKD and
Post-Quantum Cryptography Meet" (Rane and Orthi, 2026).

## Files

| File | Purpose | Source |
|---|---|---|
| `simulation_complete.py` | Main simulation: Tables II–IV, Figs 3–7 | Already in the Zenodo record |
| `fp_validation.py` | F2-bis empirical FP validation at optimised thresholds (Table VI) | **NEW** in v1.3 |
| `wilson_ci.py` | Standalone 95% Wilson CI helper | **NEW** in v1.3 |
| `requirements.txt` | Pinned Python dependencies | **NEW** in v1.3 |

> Note: `simulation_complete.py` lives in the upstream Zenodo record
> (DOI: 10.5281/zenodo.20149707). Copy it into this directory before running
> `fp_validation.py`, which imports from it.

## Environment

- Python 3.13 (verified on macOS 14, Apple Silicon)
- All dependencies pure-Python or pip-installable; no system packages required.

## Setup

```bash
python3 -m venv .venv-qkd
source .venv-qkd/bin/activate          # Windows: .venv-qkd\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the simulations

### Main results (Tables II–IV, Figures 3–7)

```bash
python simulation_complete.py
```

Random seed 42 is hard-coded. Output files (`sim_table*.csv`, `fig*.png`) are
written to the current directory. Expected runtime: under one minute on a modern
laptop.

### F2-bis empirical FP validation (Table VI)

```bash
python fp_validation.py
```

This script imports `single_trial()` from `simulation_complete.py`, overrides
the CHSH and QBER abort thresholds to the optimised operating point, and runs
100,000 independent trials at p_eve = 0 to measure the empirical false-positive
rate. Expected runtime: under one second.

### Wilson CI helper (standalone)

```bash
python wilson_ci.py 392 100000
```

Computes a 95% Wilson confidence interval on any binomial proportion. Useful
for converting any future false-positive count into a publishable CI.

## Expected output — F2-bis (paper-of-record reference values)

```
single_trial() returns a 6-tuple; boolean fields at positions [3, 4, 5]

Per-position True counts (out of 100,000):
  Position [3] :    392  (0.3920%)
  Position [4] :     16  (0.0160%)
  Position [5] :    376  (0.3760%)
Combined OR (any abort):    392  (0.3920%)

Empirical FP (combined): 0.3920%
95% Wilson CI          : [0.3551%, 0.4327%]
Analytical prediction  : 0.4030%

Analytical within empirical 95% CI: YES
```

These values are reported in Table VI of the v1.3 manuscript. The 16-count
CHSH-only fraction (0.016%) matches the predicted ~0.023% Gaussian-tail
contribution at the |S| < 2.66 threshold for mean = 2.80, σ ≈ 0.04. The 376
QBER-only count dominates, as expected at this operating point. No trial
triggered both abort conditions simultaneously.

## Citing

If you re-use these scripts, cite both:

1. Rane, V. and Orthi, S. M. (2026). *Two Roads to Quantum Safety: Where QKD
   and Post-Quantum Cryptography Meet.* [Manuscript].
2. Orthi, S. M. and Rane, V. D. (2026). *Simulation Code: Two Roads to
   Quantum Safety — Hybrid E91 QKD + ML-DSA.* Zenodo.
   DOI: 10.5281/zenodo.20149707
