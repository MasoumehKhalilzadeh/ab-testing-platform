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

    This is the automated version of the manual calculation we did by hand:
    wobble -> z-score -> p-value -> significant or not.
    """
    # Step 1: observed conversion rates (e.g. 0.125 for 12.5%)
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n

    # Step 2: the observed gap between the two groups
    observed_gap = treatment_rate - control_rate

        # Step 3: "wobble" for each group individually
    # (how much random variation we'd expect in each group's rate)
    se_control = np.sqrt(control_rate * (1 - control_rate) / control_n)
    se_treatment = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_n)

    # Step 4: combine the two wobbles into one "combined wobble" for the gap
    standard_error = np.sqrt(se_control**2 + se_treatment**2)

        # Step 5: z-score = how many "wobbles" away from zero the gap is
    z_score = observed_gap / standard_error

    # Step 6: convert z-score into a p-value using the normal distribution
    # (two-sided test: we care about the gap in either direction)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Step 7: is the result significant? (compare p-value to our threshold, e.g. 0.05)
    is_significant = p_value < alpha

    # Step 8: confidence interval for the gap
    # This gives a RANGE we're confident the true effect falls within,
    # not just a single number. Uses the same "wobble" from before.
    z_critical = stats.norm.ppf(1 - alpha / 2)  # e.g. 1.96 for 95% confidence
    margin_of_error = z_critical * standard_error
    confidence_interval = (
        observed_gap - margin_of_error,
        observed_gap + margin_of_error,
    )

    # Step 9: package everything into our result object
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

if __name__ == "__main__":
    # Quick manual test using the exact numbers from our synthetic data
    # (control: 12.5% of 9938, treatment: 14.6% of 10062)
    result = two_proportion_z_test(
        control_conversions=1240,      # approx 12.5% of 9938
        control_n=9938,
        treatment_conversions=1470,    # approx 14.6% of 10062
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

    