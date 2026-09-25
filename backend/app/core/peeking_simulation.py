"""
Peeking simulation: demonstrates why checking A/B test results every day
(instead of waiting for your planned sample size) inflates the false
positive rate far beyond the 5% you think you're accepting.

We simulate many NULL experiments (no true effect) and check, for each one,
whether "peeking" every day would have falsely declared significance at
SOME point during the experiment.
"""

import numpy as np
from scipy import stats

import sys
import os
sys.path.append(os.path.dirname(__file__))
from stats_engine import two_proportion_z_test

def simulate_one_experiment_with_peeking(
    daily_users_per_group: int = 50,
    num_days: int = 30,
    true_conversion_rate: float = 0.12,
    seed: int = None,
) -> bool:
    """
    Simulates a NULL experiment (both groups have the SAME true conversion
    rate — i.e. the treatment does nothing) where someone checks the
    z-test result every day and stops as soon as p < 0.05.

    Returns True if this "peeking" strategy would have falsely declared
    significance at some point (a false positive), False otherwise.
    """
    rng = np.random.default_rng(seed)

    control_conversions = 0
    control_n = 0
    treatment_conversions = 0
    treatment_n = 0

    for day in range(num_days):
        # Step 1: simulate today's new users in each group
        # (both groups have the SAME true rate, since this is a null experiment)
        new_control = rng.binomial(daily_users_per_group, true_conversion_rate)
        new_treatment = rng.binomial(daily_users_per_group, true_conversion_rate)

        # Step 2: accumulate totals (like a running total, day after day)
        control_conversions += new_control
        control_n += daily_users_per_group
        treatment_conversions += new_treatment
        treatment_n += daily_users_per_group

        # Step 3: check significance TODAY (this is the "peeking" behavior)
        result = two_proportion_z_test(
            control_conversions, control_n,
            treatment_conversions, treatment_n,
        )

        if result.is_significant:
            return True  # falsely declared significant at some point!

    return False  # never falsely triggered during the whole experiment

if __name__ == "__main__":
    NUM_SIMULATIONS = 2000
    false_positives = 0

    for i in range(NUM_SIMULATIONS):
        result = simulate_one_experiment_with_peeking(seed=i)
        if result:
            false_positives += 1

    false_positive_rate = false_positives / NUM_SIMULATIONS

    print(f"Number of simulated null experiments: {NUM_SIMULATIONS}")
    print(f"False positives (falsely called 'significant'): {false_positives}")
    print(f"Observed false positive rate: {false_positive_rate:.2%}")
    print(f"Expected false positive rate (if checked ONLY once, at the end): 5.00%")