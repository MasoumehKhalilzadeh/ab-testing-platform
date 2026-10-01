"""
CUPED (Controlled-experiment Using Pre-Experiment Data).

The idea: some of the "noise" in our experiment outcomes is actually
predictable from data we already had BEFORE the experiment started
(e.g. a user's historical spending). By removing that predictable part,
we reduce the wobble in our measurements -- letting us detect real
effects with fewer users or less time.

Used in production at Netflix, Microsoft, Booking.com, and others.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_generator import generate_experiment_data, ExperimentConfig
from stats_engine import two_sample_t_test


def apply_cuped(
    outcome: np.ndarray,
    pre_experiment_covariate: np.ndarray,
) -> np.ndarray:
    """
    Adjusts the outcome variable using CUPED, reducing its variance by
    removing the part explainable by the pre-experiment covariate.

    Returns the adjusted outcome (same length as input), which should
    have LOWER variance than the original, while preserving the same
    average treatment effect.
    """
    # Step 1: calculate theta -- how strongly the pre-experiment covariate
    # predicts the outcome. This is the same formula used in linear regression:
    # theta = covariance(outcome, covariate) / variance(covariate)
    covariance_matrix = np.cov(outcome, pre_experiment_covariate)
    covariance = covariance_matrix[0, 1]
    variance_covariate = covariance_matrix[1, 1]
    theta = covariance / variance_covariate

    # Step 2: subtract out the predictable part of the outcome
    # (how far each person's pre-experiment value was from the average,
    # scaled by theta)
    covariate_mean = np.mean(pre_experiment_covariate)
    adjusted_outcome = outcome - theta * (pre_experiment_covariate - covariate_mean)

    return adjusted_outcome


if __name__ == "__main__":
    # Generate synthetic data with a real revenue lift injected,
    # and a STRONGER correlation between pre-experiment metric and outcome
    # (closer to what a well-chosen covariate looks like in practice)
    df = generate_experiment_data(
        ExperimentConfig(
            true_lift_conversion=0.02,
            true_lift_revenue=1.5,
            pre_experiment_corr=0.9,
        )
    )

    control_df = df[df["group"] == "control"]
    treatment_df = df[df["group"] == "treatment"]

    # --- WITHOUT CUPED: run the t-test on raw revenue ---
    raw_result = two_sample_t_test(
        control_df["revenue"].values,
        treatment_df["revenue"].values,
    )

    # --- WITH CUPED: adjust revenue using pre-experiment data, then t-test ---
    control_adjusted = apply_cuped(
        control_df["revenue"].values,
        control_df["pre_experiment_metric"].values,
    )
    treatment_adjusted = apply_cuped(
        treatment_df["revenue"].values,
        treatment_df["pre_experiment_metric"].values,
    )
    cuped_result = two_sample_t_test(control_adjusted, treatment_adjusted)

    print("--- WITHOUT CUPED ---")
    print(f"Observed gap:     {raw_result.observed_gap:.4f}")
    print(f"Standard error:   {raw_result.standard_error:.4f}")
    print(f"T-score:          {raw_result.t_score:.2f}")
    print(f"P-value:          {raw_result.p_value:.8f}")

    print("\n--- WITH CUPED ---")
    print(f"Observed gap:     {cuped_result.observed_gap:.4f}")
    print(f"Standard error:   {cuped_result.standard_error:.4f}")
    print(f"T-score:          {cuped_result.t_score:.2f}")
    print(f"P-value:          {cuped_result.p_value:.8f}")

    variance_reduction = 1 - (cuped_result.standard_error / raw_result.standard_error) ** 2