#!/usr/bin/env python3
"""
F2-bis: Empirical false-positive validation at optimised abort thresholds.

This script imports the primitives from `simulation_complete.py` (without
modifying it), overrides the CHSH and QBER abort thresholds to the optimised
operating point (CHSH = 2.66, QBER = 2.33%), and runs N_TRIALS = 100,000
independent single-trial simulations at p_eve = 0 to measure the empirical
false-positive abort rate.

Outputs:
    - Per-position True counts on every boolean field returned by single_trial()
    - Combined OR (any-abort) count
    - 95% Wilson confidence interval on the combined OR
    - Pass/fail check whether the analytical prediction falls inside the CI

Reproducibility:
    Random seed = 42 (same as published simulation).
"""
import sys
import time
from math import sqrt

import numpy as np

sys.path.insert(0, '.')
import simulation_complete as sim

# Override thresholds in-memory (does NOT modify simulation_complete.py)
sim.CHSH_THRESH = 2.66
sim.QBER_THRESH = 0.0233

# Configuration
N_TRIALS = 100_000
DIST_KM = 10  # FP rate is essentially distance-independent at p = 0
P_EVE = 0.0   # no eavesdropping — FP = P(abort | no Eve)


def wilson_ci(k, n, alpha=0.05):
    """Two-sided 95% Wilson confidence interval for a binomial proportion."""
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    halfw = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - halfw, centre + halfw


def main():
    # Step 1: inspect single_trial's return shape
    np.random.seed(sim.SEED)
    sample = sim.single_trial(DIST_KM, P_EVE)

    print("=" * 72)
    print("F2-bis: empirical FP validation at optimised thresholds")
    print("=" * 72)
    print(f"  CHSH threshold  : {sim.CHSH_THRESH}")
    print(f"  QBER threshold  : {sim.QBER_THRESH * 100:.2f}%")
    print(f"  Distance        : {DIST_KM} km")
    print(f"  Eavesdropping p : {P_EVE}")
    print(f"  N_TRIALS        : {N_TRIALS:,}")
    print(f"  Random seed     : {sim.SEED}")
    print()

    bool_indices = [
        i for i, v in enumerate(sample)
        if isinstance(v, (bool, np.bool_))
    ]
    print(f"single_trial() returns a {len(sample)}-tuple; "
          f"boolean fields at positions {bool_indices}")
    print()

    # Step 2: run N_TRIALS independent trials, count True at each boolean field
    np.random.seed(sim.SEED)
    counts = {i: 0 for i in bool_indices}
    n_any = 0  # combined OR across all boolean fields

    t_start = time.time()
    print("Running trials...")
    for trial in range(N_TRIALS):
        r = sim.single_trial(DIST_KM, P_EVE)
        any_abort = False
        for i in bool_indices:
            if bool(r[i]):
                counts[i] += 1
                any_abort = True
        if any_abort:
            n_any += 1
        if (trial + 1) % 20_000 == 0:
            elapsed = time.time() - t_start
            print(f"  {trial + 1:>7,} / {N_TRIALS:,}   "
                  f"elapsed {elapsed:.1f}s   "
                  f"any-abort so far = {n_any / (trial + 1) * 100:.4f}%")

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total:.1f}s")
    print()

    # Step 3: report
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Per-position True counts (out of {N_TRIALS:,}):")
    for i in bool_indices:
        pct = 100 * counts[i] / N_TRIALS
        print(f"    Position [{i}] : {counts[i]:>6,d}  ({pct:.4f}%)")
    print(f"  Combined OR (any abort): {n_any:>6,d}  ({100 * n_any / N_TRIALS:.4f}%)")
    print()

    fp_rate = n_any / N_TRIALS
    lo, hi = wilson_ci(n_any, N_TRIALS)
    analytical = 0.00403  # threshold-grid evaluation at optimised operating point

    print(f"  Empirical FP (combined): {fp_rate * 100:.4f}%")
    print(f"  95% Wilson CI          : [{lo * 100:.4f}%, {hi * 100:.4f}%]")
    print(f"  Analytical prediction  : {analytical * 100:.4f}%")
    print()

    in_ci = lo <= analytical <= hi
    print(f"  Analytical within empirical 95% CI: {'YES' if in_ci else 'NO'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
