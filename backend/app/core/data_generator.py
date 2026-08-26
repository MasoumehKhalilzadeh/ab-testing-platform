"""
Synthetic A/B test data generator.

Why this exists:
Real experiment datasets don't tell you the "true" effect size — you only ever
see noisy observed outcomes. To validate that a stats engine is correct (controls
false positive rate, has adequate power, etc.), you need data where YOU know the
ground truth. This generator lets us do that.

We simulate a mobile-app style experiment:
- Users are randomized into control / treatment
- Each user has a pre-experiment covariate (e.g. historical engagement score) —
  this is what CUPED will later use for variance reduction
- Each user has a binary outcome (e.g. did they convert / retain) and a
  continuous outcome (e.g. revenue or session time)
- Treatment can have a configurable true effect on both outcomes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    n_users: int = 20_000
    baseline_conversion_rate: float = 0.12       # e.g. trial-to-paid conversion
    true_lift_conversion: float = 0.0            # absolute lift, e.g. 0.01 = +1pp
    baseline_revenue_mean: float = 5.0           # e.g. avg revenue per user ($)
    revenue_std: float = 8.0
    true_lift_revenue: float = 0.0               # absolute $ lift in treatment
    pre_experiment_corr: float = 0.5             # correlation of covariate with outcome (for CUPED)
    seed: int = 42


def generate_experiment_data(config: ExperimentConfig) -> pd.DataFrame:
    """
    Returns a user-level DataFrame with columns:
        user_id, group ('control'/'treatment'), pre_experiment_metric,
        converted (0/1), revenue (float)
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_users

    # 1. Random assignment (50/50 split)
    group = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])
    is_treatment = (group == "treatment").astype(int)

    # 2. Pre-experiment covariate: correlated with the user's "latent quality"
    #    This simulates something like "average sessions in prior 30 days"
    latent_quality = rng.normal(0, 1, size=n)
    pre_experiment_metric = (
        config.pre_experiment_corr * latent_quality
        + np.sqrt(1 - config.pre_experiment_corr**2) * rng.normal(0, 1, size=n)
    )
    pre_experiment_metric = 10 + 3 * pre_experiment_metric
    pre_experiment_metric = np.clip(pre_experiment_metric, 0, None)

    # 3. Binary outcome (conversion), driven by latent quality + treatment effect
    base_logit = np.log(config.baseline_conversion_rate / (1 - config.baseline_conversion_rate))
    logit = base_logit + 0.4 * latent_quality + is_treatment * (
        np.log((config.baseline_conversion_rate + config.true_lift_conversion + 1e-9) /
                (1 - config.baseline_conversion_rate - config.true_lift_conversion + 1e-9))
        - base_logit
    )
    prob_convert = 1 / (1 + np.exp(-logit))
    converted = rng.binomial(1, prob_convert)

    # 4. Continuous outcome (revenue), driven by latent quality + treatment effect
    revenue_mean = (
        config.baseline_revenue_mean
        + 2.0 * latent_quality
        + is_treatment * config.true_lift_revenue
    )
    revenue = rng.normal(revenue_mean, config.revenue_std)
    revenue = np.clip(revenue, 0, None)

    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "group": group,
        "pre_experiment_metric": pre_experiment_metric,
        "converted": converted,
        "revenue": revenue,
    })
    return df


if __name__ == "__main__":
    # Quick smoke test: generate a null experiment (no true effect) and a
    # true-effect experiment, and sanity check the observed rates.
    null_cfg = ExperimentConfig(true_lift_conversion=0.0, true_lift_revenue=0.0)
    effect_cfg = ExperimentConfig(true_lift_conversion=0.02, true_lift_revenue=1.5)

    for name, cfg in [("NULL effect", null_cfg), ("TRUE effect", effect_cfg)]:
        df = generate_experiment_data(cfg)
        summary = df.groupby("group").agg(
            n=("user_id", "count"),
            conv_rate=("converted", "mean"),
            avg_revenue=("revenue", "mean"),
        )
        print(f"\n--- {name} ---")
        print(summary)