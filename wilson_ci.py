#!/usr/bin/env python3
"""
95% Wilson confidence interval for a binomial proportion.

Usage:
    python wilson_ci.py <k> <n>

Example:
    python wilson_ci.py 392 100000
    -> k = 392, n = 100,000
       Empirical proportion: 0.3920%
       95% Wilson CI:        [0.3551%, 0.4327%]
"""
import sys
from math import sqrt


def wilson_ci(k, n, alpha=0.05):
    """Two-sided 95% Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes (events).
    n : int
        Total number of trials.
    alpha : float, default=0.05
        Significance level (alpha = 0.05 -> 95% CI).

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds of the confidence interval.
    """
    z = 1.959963984540054  # scipy.stats.norm.ppf(1 - alpha/2) for alpha=0.05
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    halfw = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - halfw, centre + halfw


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python wilson_ci.py <k> <n>", file=sys.stderr)
        sys.exit(1)
    k, n = int(sys.argv[1]), int(sys.argv[2])
    lo, hi = wilson_ci(k, n)
    print(f"k = {k:,}, n = {n:,}")
    print(f"Empirical proportion: {100 * k / n:.4f}%")
    print(f"95% Wilson CI:        [{100 * lo:.4f}%, {100 * hi:.4f}%]")
