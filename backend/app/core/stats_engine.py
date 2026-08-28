"""
Stats engine: core hypothesis testing functions for A/B tests.

This module answers the central question: "is the observed difference
between control and treatment real, or could it be explained by random
chance (wobble)?"
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_generator import generate_experiment_data, ExperimentConfig


@dataclass
class ProportionTestResult:
    control_rate: float
    treatment_rate: float
    observed_gap: float
    standard_error: float
    z_score: float
    p_value: float
    is_significant: bool
    confidence_interval: tuple


def two_proportion_z_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ProportionTestResult:
    """
    Compares conversion rates between two groups using a z-test.
    """
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    observed_gap = treatment_rate - control_rate

    se_control = np.sqrt(control_rate * (1 - control_rate) / control_n)
    se_treatment = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_n)
    standard_error = np.sqrt(se_control**2 + se_treatment**2)

    z_score = observed_gap / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    is_significant = p_value < alpha

    z_critical = stats.norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * standard_error
    confidence_interval = (
        observed_gap - margin_of_error,
        observed_gap + margin_of_error,
    )

    return ProportionTestResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        observed_gap=observed_gap,
        standard_error=standard_error,
        z_score=z_score,
        p_value=p_value,
        is_significant=is_significant,
        confidence_interval=confidence_interval,
    )

@dataclass
class MeansTestResult:
    control_mean: float
    treatment_mean: float
    observed_gap: float
    standard_error: float
    t_score: float
    p_value: float
    is_significant: bool
    confidence_interval: tuple


def two_sample_t_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
) -> MeansTestResult:
    """
    Compares average values (e.g. revenue) between two groups using a t-test.
    Same idea as the z-test, but for continuous numbers instead of percentages.
    """
    # Step 1: basic averages and the gap between them
    control_mean = np.mean(control_values)
    treatment_mean = np.mean(treatment_values)
    observed_gap = treatment_mean - control_mean

    # Step 2: run the t-test using scipy's built-in function
    # (this does the standard deviation, wobble, and t-score calculations for us)
    t_score, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)

    # Step 3: is it significant?
    is_significant = p_value < alpha

    # Step 4: standard error, needed for the confidence interval
    se_control = np.std(control_values, ddof=1) / np.sqrt(len(control_values))
    se_treatment = np.std(treatment_values, ddof=1) / np.sqrt(len(treatment_values))
    standard_error = np.sqrt(se_control**2 + se_treatment**2)

    # Step 5: confidence interval (same idea as before)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    margin_of_error = z_critical * standard_error
    confidence_interval = (
        observed_gap - margin_of_error,
        observed_gap + margin_of_error,
    )

    return MeansTestResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        observed_gap=observed_gap,
        standard_error=standard_error,
        t_score=t_score,
        p_value=p_value,
        is_significant=is_significant,
        confidence_interval=confidence_interval,
    )

def sample_size_for_proportion(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculates how many users PER GROUP are needed to reliably detect
    a given effect size, at a given confidence and power level.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    pooled_variance = p1 * (1 - p1) + p2 * (1 - p2)

    n = ((z_alpha + z_power) ** 2 * pooled_variance) / (minimum_detectable_effect ** 2)
    return int(np.ceil(n))


if __name__ == "__main__":
    result = two_proportion_z_test(
        control_conversions=1240,
        control_n=9938,
        treatment_conversions=1470,
        treatment_n=10062,
    )

    print(f"Control rate:       {result.control_rate:.4f}")
    print(f"Treatment rate:     {result.treatment_rate:.4f}")
    print(f"Observed gap:       {result.observed_gap:.4f}")
    print(f"Standard error:     {result.standard_error:.4f}")
    print(f"Z-score:            {result.z_score:.2f}")
    print(f"P-value:            {result.p_value:.6f}")
    print(f"Significant?:       {result.is_significant}")
    print(f"95% CI:             ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")

    print("\n--- Sample Size Calculator ---")
    required_n = sample_size_for_proportion(
        baseline_rate=0.12,
        minimum_detectable_effect=0.02,
    )
    print(f"Required sample size per group: {required_n}")


    print("\n--- T-Test on Synthetic Revenue Data ---")
    df = generate_experiment_data(
        ExperimentConfig(true_lift_conversion=0.02, true_lift_revenue=1.5)
    )
    control_revenue = df[df["group"] == "control"]["revenue"].values
    treatment_revenue = df[df["group"] == "treatment"]["revenue"].values

    t_result = two_sample_t_test(control_revenue, treatment_revenue)

    print(f"Control avg revenue:    {t_result.control_mean:.4f}")
    print(f"Treatment avg revenue:  {t_result.treatment_mean:.4f}")
    print(f"Observed gap:           {t_result.observed_gap:.4f}")
    print(f"Standard error:         {t_result.standard_error:.4f}")
    print(f"T-score:                {t_result.t_score:.2f}")
    print(f"P-value:                {t_result.p_value:.6f}")
    print(f"Significant?:           {t_result.is_significant}")
    print(f"95% CI:                 ({t_result.confidence_interval[0]:.4f}, {t_result.confidence_interval[1]:.4f})")