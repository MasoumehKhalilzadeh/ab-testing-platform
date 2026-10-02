"""
Lift — Automated A/B Testing & Experiment Analysis Platform
Simple dashboard tying together: synthetic data generation, statistical
testing (z-test/t-test), CUPED, and AI-generated recommendations.
"""

import streamlit as st
import sys
import os

# Let this file import our backend code
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend", "app", "core"))

from data_generator import generate_experiment_data, ExperimentConfig
from stats_engine import two_proportion_z_test, two_sample_t_test
from cuped import apply_cuped
from ai_interpreter import interpret_experiment_results

st.set_page_config(page_title="Lift — A/B Testing Platform", layout="centered")

st.title("📊 Lift")
st.caption("Automated A/B Testing & Experiment Analysis Platform")

st.header("1. Generate Experiment Data")

n_users = st.slider("Number of users", 1000, 50000, 20000, step=1000)
true_lift_conversion = st.slider("True conversion lift (for simulation)", 0.0, 0.05, 0.02, step=0.005)
true_lift_revenue = st.slider("True revenue lift (for simulation, $)", 0.0, 5.0, 1.5, step=0.1)

if st.button("Generate & Analyze Experiment"):
    st.session_state["run"] = True

if st.session_state.get("run"):
    config = ExperimentConfig(
        n_users=n_users,
        true_lift_conversion=true_lift_conversion,
        true_lift_revenue=true_lift_revenue,
    )
    df = generate_experiment_data(config)

    control_df = df[df["group"] == "control"]
    treatment_df = df[df["group"] == "treatment"]

    st.header("2. Conversion Rate Results")

    z_result = two_proportion_z_test(
        control_conversions=int(control_df["converted"].sum()),
        control_n=len(control_df),
        treatment_conversions=int(treatment_df["converted"].sum()),
        treatment_n=len(treatment_df),
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Control Rate", f"{z_result.control_rate:.2%}")
    col2.metric("Treatment Rate", f"{z_result.treatment_rate:.2%}")
    col3.metric("P-value", f"{z_result.p_value:.5f}")

    if z_result.is_significant:
        st.success(f"✅ Statistically significant (gap: {z_result.observed_gap:.2%})")
    else:
        st.warning(f"⚠️ Not statistically significant (gap: {z_result.observed_gap:.2%})")

    st.header("3. Revenue Results")

    t_result = two_sample_t_test(
        control_df["revenue"].values,
        treatment_df["revenue"].values,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Control Avg Revenue", f"${t_result.control_mean:.2f}")
    col2.metric("Treatment Avg Revenue", f"${t_result.treatment_mean:.2f}")
    col3.metric("P-value", f"{t_result.p_value:.5f}")

    if t_result.is_significant:
        st.success(f"✅ Statistically significant (gap: ${t_result.observed_gap:.2f})")
    else:
        st.warning(f"⚠️ Not statistically significant (gap: ${t_result.observed_gap:.2f})")

        st.header("4. AI-Generated Recommendation")

    with st.spinner("Asking Claude to interpret the results..."):
        summary = interpret_experiment_results(
            metric_name="Revenue per user",
            control_value=t_result.control_mean,
            treatment_value=t_result.treatment_mean,
            p_value=t_result.p_value,
            confidence_interval=t_result.confidence_interval,
            is_significant=t_result.is_significant,
        )

    st.info(summary.replace("$", "\\$"))