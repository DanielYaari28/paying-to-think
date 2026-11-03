"""
Paying to Think: POWER ANALYSIS (sample-size justification)

Reproduces the pre-registered power analysis from the pilot accuracies.
CPU only; runs in about a second.

Method (course notes, Chapter 4, "Method 1: Pilot Study"):
    - Treat correct/incorrect as a 0/1 outcome.
    - Cohen's d for two accuracies p1, p2 with pooled SD:
          d = |p1 - p2| / sqrt((p1(1 - p1) + p2(1 - p2)) / 2)
    - statsmodels TTestIndPower, power = 0.80, two-sided,
      alpha = 0.05 / 3 = 0.0167 (Bonferroni, so each of the three
      pairwise post hoc tests is itself powered).
    - The largest required n (the smallest d) sets the sample size.

Input:  data/pilot/pilot2_summary.csv  (written by scripts/pilot/pilot_part2.py)
        If that file is missing, the pre-registered pilot accuracies are used.
Output: results/power_analysis.csv

Usage (from the repo root):
    python scripts/analysis/power_analysis.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

PILOT_SUMMARY = os.path.join("data", "pilot", "pilot2_summary.csv")
OUT = os.path.join("results", "power_analysis.csv")

ALPHA, POWER, N_COMPARISONS = 0.05, 0.80, 3
ALPHA_ADJ = ALPHA / N_COMPARISONS
N_CHOSEN = 350                      # pre-registered sample size per arm
PAIRS = [("low", "unlimited"), ("medium", "unlimited"), ("low", "medium")]

# Pre-registered pilot accuracies (40/60, 47/60, 53/60), used if the CSV is absent
PILOT_FALLBACK = {"low": 40 / 60, "medium": 47 / 60, "unlimited": 53 / 60}


def load_pilot_accuracy():
    if os.path.exists(PILOT_SUMMARY):
        acc = pd.read_csv(PILOT_SUMMARY, index_col="arm")["accuracy"].to_dict()
        print(f"Pilot accuracies read from {PILOT_SUMMARY}")
    else:
        acc = PILOT_FALLBACK
        print(f"{PILOT_SUMMARY} not found; using the pre-registered pilot accuracies")
    return acc


def cohens_d(p1, p2):
    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return abs(p1 - p2) / pooled_sd


def n_by_formula(d, alpha, power):
    """Normal-approximation check: n = 2 (z_{1-alpha/2} + z_{1-beta})^2 / d^2."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return 2 * (z_a + z_b) ** 2 / d ** 2


def main():
    acc = load_pilot_accuracy()
    print(f"low = {acc['low']:.3f}, medium = {acc['medium']:.3f}, unlimited = {acc['unlimited']:.3f}\n")

    solver = TTestIndPower()
    rows = []
    for a, b in PAIRS:
        d = cohens_d(acc[a], acc[b])
        n = solver.solve_power(effect_size=d, alpha=ALPHA_ADJ, power=POWER, alternative="two-sided")
        power_at_chosen = solver.power(effect_size=d, nobs1=N_CHOSEN, alpha=ALPHA_ADJ, alternative="two-sided")
        rows.append(dict(comparison=f"{a} vs {b}", acc_a=acc[a], acc_b=acc[b], cohens_d=d,
                         n_per_arm_statsmodels=n, n_per_arm_formula=n_by_formula(d, ALPHA_ADJ, POWER),
                         power_at_350=power_at_chosen))

    table = pd.DataFrame(rows)
    print(f"Power = {POWER}, alpha = {ALPHA}/{N_COMPARISONS} = {ALPHA_ADJ:.4f} (Bonferroni), two-sided")
    for r in table.itertuples():
        print(f"  {r.comparison:<20} d = {r.cohens_d:.3f} | n per arm = {np.ceil(r.n_per_arm_statsmodels):.0f} "
              f"(formula check {r.n_per_arm_formula:.0f}) | power at n = {N_CHOSEN}: {r.power_at_350:.2f}")

    binding = table.loc[table["n_per_arm_statsmodels"].idxmax()]
    print(f"\nLargest requirement: {np.ceil(binding.n_per_arm_statsmodels):.0f} per arm ({binding.comparison}).")
    print(f"Chosen: {N_CHOSEN} per arm ({3 * N_CHOSEN:,} total) as a buffer for imprecise pilot estimates.")

    n_unadjusted = solver.solve_power(effect_size=binding.cohens_d, alpha=ALPHA, power=POWER,
                                      alternative="two-sided")
    print(f"For comparison, without the Bonferroni correction (alpha = 0.05): "
          f"{np.ceil(n_unadjusted):.0f} per arm.")

    os.makedirs("results", exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
