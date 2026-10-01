"""
AI interpretation layer: takes statistical results and uses Claude to
generate a plain-English summary and recommendation (ship / kill / wait).

This is the "AI" piece of the AI Experimentation Platform -- it turns
precise but hard-to-read statistics into something a non-technical
stakeholder (like a product manager) can act on immediately.
"""

import os
from dotenv import load_dotenv
import anthropic

# Load the API key from our .env file (never hardcoded directly in code)
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def interpret_experiment_results(
    metric_name: str,
    control_value: float,
    treatment_value: float,
    p_value: float,
    confidence_interval: tuple,
    is_significant: bool,
) -> str:
    """
    Sends experiment statistics to Claude and returns a plain-English
    summary + recommendation (ship / kill / wait).
    """
    # Step 1: build a clear, structured prompt describing our results
    prompt = f"""You are a data science assistant helping a product manager
understand an A/B test result. Here are the statistics:

Metric: {metric_name}
Control value: {control_value:.4f}
Treatment value: {treatment_value:.4f}
P-value: {p_value:.6f}
95% Confidence interval for the effect: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})
Statistically significant (alpha=0.05): {is_significant}

In 3-4 short sentences, written in plain English for a non-technical
audience:
1. Explain what happened in the experiment.
2. State whether the result is statistically significant.
3. Give a clear recommendation: "ship it", "do not ship", or "collect more data" -- and briefly say why.
"""

    # Step 2: send the prompt to Claude's API
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    # Step 3: extract and return the text response
    return response.content[0].text


if __name__ == "__main__":
    # Quick test using our earlier z-test result (conversion rate)
    summary = interpret_experiment_results(
        metric_name="Conversion rate",
        control_value=0.1248,
        treatment_value=0.1461,
        p_value=0.00001,
        confidence_interval=(0.0118, 0.0308),
        is_significant=True,
    )

    print("--- AI-Generated Recommendation ---")
    print(summary)