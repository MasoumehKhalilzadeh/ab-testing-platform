# 📊 Lift — Automated A/B Testing & Experiment Analysis Platform

A statistically rigorous, end-to-end A/B testing platform that automates the full experimentation lifecycle: from sample size planning, to significance testing, to variance reduction, to plain-English AI-generated recommendations.

Built to explore (and prove, not just implement) the statistical methods real companies use to decide whether a product change actually works — or just *looks* like it does.

---

## Why I Built This

Most "A/B testing" portfolio projects run a single `scipy.stats.ttest_ind()` call on a toy dataset and call it done. I wanted to go deeper: understand *why* each statistical method works, where naive approaches break down in practice, and how production experimentation platforms (think Netflix, Microsoft, Booking.com) actually address those failure modes.

So instead of just implementing textbook formulas, I used them to **prove things**:

- I proved that checking experiment results daily — a mistake almost every team makes at some point — inflates the false positive rate from an intended 5% to **~30%**, using a 2,000-trial Monte Carlo simulation on synthetic null experiments.
- I then implemented a correction and proved it brings the false positive rate back down to a controlled level.
- I implemented CUPED, a variance-reduction technique used in production experimentation systems, and measured its actual impact on detection power under different covariate-correlation strengths — rather than just dropping in the formula and assuming it works.

This project is less "I used statistics" and more "I verified my statistics are actually trustworthy before trusting them."

---

## What It Does

1. **Synthetic data generation** — generates realistic user-level experiment data (conversion events + revenue) with *configurable, known ground-truth effects*, so every statistical method built on top of it can be validated against a known answer instead of taken on faith.
2. **Hypothesis testing** — two-proportion z-test (conversion rates) and two-sample t-test (revenue/continuous metrics), with p-values and 95% confidence intervals.
3. **Sample size & power calculation** — tells you how many users you need *before* launching an experiment, based on your minimum detectable effect, baseline rate, and desired power.
4. **Sequential testing / peeking correction** — demonstrates the false-positive inflation caused by repeated significance checks, and corrects for it using a Bonferroni-adjusted threshold.
5. **CUPED (Controlled-experiment Using Pre-Experiment Data)** — reduces variance in the outcome metric using pre-experiment covariates, increasing statistical power without needing more users.
6. **AI-generated recommendations** — feeds the statistical output into Claude (Anthropic's LLM) to produce a plain-English summary and a clear ship / don't-ship / collect-more-data recommendation for non-technical stakeholders.
7. **Interactive dashboard** — a Streamlit app that ties all of the above together into a single, explorable interface.

---

## Key Result: The Peeking Problem, Quantified

| Scenario | False Positive Rate |
|---|---|
| Checking results only once, at the planned end (textbook assumption) | 5.00% (target) |
| Checking results every day for 30 days, no correction | **29.65%** |
| Checking results every day for 30 days, with Bonferroni correction | **1.70%** |

All three numbers come from 2,000-trial simulations on data where the true effect is exactly zero, so any "significant" result is by definition a false alarm. This is the kind of mistake that has led real companies to ship changes based on noise — and the kind of thing experimentation platforms exist to prevent.

---

## Tech Stack

- **Python** — numpy, pandas, scipy for data generation and statistical computation
- **Anthropic API (Claude)** — AI-generated plain-English experiment summaries
- **Streamlit** — interactive dashboard
- **FastAPI** *(scaffolded for future API layer)*

---

## Project Structure

```
ab-testing-platform/
├── backend/
│   └── app/
│       └── core/
│           ├── data_generator.py      # Synthetic data with known ground-truth effects
│           ├── stats_engine.py        # Z-test, t-test, sample size/power calculator
│           ├── peeking_simulation.py  # Monte Carlo proof of the peeking problem + fix
│           ├── cuped.py               # CUPED variance reduction
│           └── ai_interpreter.py      # Claude-powered plain-English recommendations
├── frontend/
│   └── dashboard.py                   # Streamlit dashboard
├── data/
└── requirements.txt
```

---

## Running It Locally

```bash
# Clone the repo
git clone https://github.com/MasoumehKhalilzadeh/ab-testing-platform.git
cd ab-testing-platform

# Set up the environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run the dashboard
streamlit run frontend/dashboard.py
```

---

## What I'd Build Next

- Alpha-spending / always-valid sequential testing (less conservative than Bonferroni, used in modern platforms like Optimizely)
- Multiple-metric correction (Benjamini-Hochberg) for experiments tracking several outcomes at once
- A persistence layer (Postgres) so experiments can be saved, revisited, and monitored over time rather than regenerated each run
- Heterogeneous treatment effect analysis (does the effect differ across user segments?)

---

## Background

I work in data analytics and built this project to deepen my understanding of experimentation methodology ahead of applying for data science / ML engineering roles. Every statistical claim in this README is backed by a simulation in the codebase — not just a formula I copied.