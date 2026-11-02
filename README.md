# Paying to Think

**Does giving a reasoning language model a larger "thinking" token budget increase its accuracy on math word problems?**

We ran DeepSeek-R1-Distill-Qwen-1.5B on 1,050 randomly sampled GSM8K test problems, randomly assigned to three thinking budgets (350 each): low (488 tokens), medium (890), and unlimited (4,096-token cap). The design, hypotheses, power analysis, and analysis plan were pre-registered on Oct 27, 2026, before the main run (see `Preregistration.md` and the commit history).

**Result:** accuracy rose from 59.4% (low) to 73.7% (medium) to 83.7% (unlimited); χ²(2, N = 1,050) = 52.06, p < .001. All three pairwise differences were significant at the Bonferroni-adjusted α = 0.0167. See `results/summary.txt` for the full output, including exploratory sensitivity checks.

## Repository layout

```
data/
  raw/assignment.csv            random assignment, saved before any problem ran
  raw/main_results.csv          raw experiment output, incl. full model text
  clean/main_results_clean.csv  validated data used for analysis (no raw text)
scripts/
  pilot/pilot_part1.py          feasibility pilot (60 GSM8K train problems)
  pilot/pilot_part2.py          sets budgets + power analysis
  analysis/clean_data.py        integrity checks + cleaned dataset
  analysis/analyze.py           pre-registered tests, exploratory checks, figures
run_main.ipynb                  main experiment (Colab, T4 GPU)
results/                        tables (CSV) + summary.txt
figures/                        figures used in the report and presentation
```

## Reproducing the analysis (any laptop, ~10 seconds)

```bash
git clone https://github.com/DanielYaari28/paying-to-think.git
cd paying-to-think
pip install -r requirements.txt
python scripts/analysis/clean_data.py
python scripts/analysis/analyze.py
```

`clean_data.py` stops with an error if any integrity check fails (wrong arm sizes, duplicates, or a result that doesn't match its pre-saved assignment). `analyze.py` writes every table to `results/` and every figure to `figures/`.

## Reproducing the experiment (needs an NVIDIA GPU)

The pilot and main run were done on Google Colab with a T4 GPU. Install `vllm`, `datasets`, and `statsmodels`, then run `scripts/pilot/pilot_part1.py`, `scripts/pilot/pilot_part2.py`, and `run_main.ipynb`. Every problem uses its dataset index as the sampling seed, and the assignment uses seed 2026, so reruns reproduce the same problems and arms. The main run took roughly 30 minutes. Outputs save to Google Drive; copy them into `data/raw/` to analyze.
