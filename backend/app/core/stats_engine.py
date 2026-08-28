"""
Stats engine: core hypothesis testing functions for A/B tests.

This module answers the central question: "is the observed difference
between control and treatment real, or could it be explained by random
chance (wobble)?"
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


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