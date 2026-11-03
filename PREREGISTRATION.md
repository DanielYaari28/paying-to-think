# Pre-registration: Paying to Think

> Markdown transcription of `docs/Preregistration.pdf`, committed to this repository on October 27, 2026, before the main experiment was run. The PDF and its commit timestamp are the authoritative record; this copy exists so the plan is readable on GitHub.

**Date:** October 27, 2026 (committed before the main experiment was run)
**Authors:** Daniel & Mihir

## Research question

Does giving a reasoning language model a larger "thinking" token budget increase its accuracy on math word problems?

## Hypotheses

- **Null (H0):** Accuracy is the same across the three thinking budgets (accuracy is independent of budget).
- **Alternative (H1):** Accuracy differs for at least one budget.
- **Significance level:** α = 0.05, chosen before data collection.

## Design

- **Model:** DeepSeek-R1-Distill-Qwen-1.5B, run with vLLM on an NVIDIA T4 GPU (float16).
- **Decoding:** temperature 0.6, top_p 0.95 (the developer's settings); random seed = the problem's dataset index, so every run is reproducible.
- **Independent variable:** thinking budget (maximum generated tokens), 3 levels:
  - **Low** = 488 tokens (25th percentile of natural generation length in the pilot)
  - **Medium** = 890 tokens (median of natural generation length in the pilot)
  - **Unlimited** = 4,096-token cap (control: the model's natural behavior)
- **Budget forcing:** if the model has not finished thinking when the budget is reached, the text is appended and the model generates up to 30 more tokens to state an answer.
- **Problems:** a random sample of 1,050 problems from the GSM8K test set (1,319 problems), drawn with seed 2026.
- **Randomization:** the 1,050 problems are shuffled (seed 2026) and assigned evenly to the three arms, 350 per arm. Each problem is used in exactly one arm, so all observations are independent.
- **Pilot separation:** all pilot problems came from the GSM8K training set and are not used in the main experiment.

## Outcome measures

- **Primary (dependent variable):** correct / incorrect. The predicted answer is extracted with a fixed rule (boxed content first, otherwise the first number after "Final answer:", lastly the last number after think) and compared numerically to the true answer. Outputs with no extractable number are counted as incorrect, and their count/arm is reported.

## Analysis plan

1. **Main test:** chi-square of independence on the 3×2 table (budget × in/correct), α = 0.05.
2. **Assumption:** independence (each problem used once); all cell counts ≥ 5 (Fisher's).
3. **Post hoc** (if the test is significant): pairwise 2 × 2 chi-square tests (low vs. medium, low vs. unlimited, medium vs. unlimited) at Bonferroni-adj α_adj = 0.05 / 3 = 0.0167.
4. **Effect sizes:** difference in accuracy between arms, and Cohen's d computed on the 0/1 outcome with pooled SD.
5. **Confidence intervals:** 95% t-based CI for each arm's accuracy and for mean tokens per arm.
6. **Interpretation rule:** a non-significant result will be reported as "failed to reject H0," never as evidence of no effect, along with the CIs.

## Power analysis

Pilot accuracies: low 0.667, medium 0.783, unlimited 0.883. Treating correct/incorrect as 0/1, Cohen's d with pooled SD, statsmodels TTestIndPower, power = 0.80, α_adj = 0.0167:

| Comparison | d | n/arm |
|---|---|---|
| Low vs. unlimited | 0.537 | 74 |
| Medium vs. unlimited | 0.271 | 287 |
| Low vs. medium | 0.264 | 303 |

The largest requirement is 303 per arm. Because pilot estimates from 60 problems are imprecise, we use 350 per arm (1,050 total) as a buffer.

## Stopping rule

All 1,050 problems will be run. Results will not be inspected before the run is complete, and the sample size will not be changed based on results.

## Exploratory (not part of the test)

In the pilot, runs that reached the 4,096-token cap appeared to be wrong more often than runs that finished on their own. This may be examined in the data and will be labeled 'exploratory'.
