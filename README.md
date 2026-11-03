# Paying to Think

**Does giving a reasoning language model a larger "thinking" token budget increase its accuracy on math word problems?**

AIPI 510 Module Project 2 · Daniel Yaari & Mihir Kosuri

We ran DeepSeek-R1-Distill-Qwen-1.5B on 1,050 randomly sampled GSM8K test problems, randomly assigned to three thinking budgets (350 per arm): low (488 tokens), medium (890), and unlimited (4,096-token cap). The hypotheses, design, power analysis, and analysis plan were preregistered on October 27, 2026, before the main run on October 30 (see `PREREGISTRATION.md` and `docs/Preregistration.pdf`; `git log --follow docs/Preregistration.pdf` shows it was first committed on October 27).

**Result:** accuracy rose from 59.4% (low) to 73.7% (medium) to 83.7% (unlimited); χ²(2, N = 1,050) = 52.06, p < .001. All three pairwise differences were significant at the Bonferroni-adjusted α = 0.0167. An exploratory sensitivity check found that most of the medium vs. unlimited gap comes from answers cut off by the budget. Full output is in `results/summary.txt`.

## Repository layout

```
PREREGISTRATION.md                  preregistered plan (readable copy of docs/Preregistration.pdf)
requirements.txt                    Python packages for the analysis scripts
data/
  raw/assignment.csv                random assignment, saved before any problem ran
  raw/main_results.csv              raw experiment output, including full model text
  clean/main_results_clean.csv      validated data used for analysis (no raw text)
scripts/
  pilot/pilot_part1.py              feasibility pilot (60 GSM8K training problems, greedy decoding)
  pilot/pilot_part2.py              pilot that set the budgets and ran the power analysis (vLLM)
  experiment/run_main.py            main experiment (1,050 test problems)
  analysis/power_analysis.py        sample-size calculation from the pilot accuracies
  analysis/clean_data.py            integrity checks + cleaned dataset
  analysis/analyze.py               preregistered tests, exploratory checks, figures
results/                            output tables (CSV) + summary.txt
figures/                            figures used in the report and presentation
docs/
  Preregistration.pdf               original timestamped preregistration
  run_main.ipynb                    Colab notebook version of the main run, with its outputs
```

## Reproducing the analysis (any laptop, no GPU)

```bash
git clone https://github.com/DanielYaari28/paying-to-think.git
cd paying-to-think
pip install -r requirements.txt
python scripts/analysis/power_analysis.py   # sample size: 303 per arm required, 350 used
python scripts/analysis/clean_data.py       # integrity checks, writes data/clean/
python scripts/analysis/analyze.py          # all tests, tables and figures
```

Run the scripts from the repository root, in this order.

- `power_analysis.py` writes `results/power_analysis.csv`. It reads the pilot accuracies from `data/pilot/pilot2_summary.csv` if that file is present, and otherwise uses the preregistered values (40/60, 47/60, 53/60).
- `clean_data.py` stops with an error if any integrity check fails: wrong arm sizes, duplicate problems, or a result that doesn't match its pre-saved assignment.
- `analyze.py` writes every table to `results/` and every figure to `figures/`.

## Reproducing the experiment (needs an NVIDIA GPU)

The pilot and main run were done on Google Colab with a T4 GPU.

```bash
pip install vllm transformers datasets pandas statsmodels
python scripts/pilot/pilot_part1.py      # feasibility pilot
python scripts/pilot/pilot_part2.py      # sets the low and medium budgets
python scripts/experiment/run_main.py    # main experiment
```

On Colab, the pilot scripts save to Google Drive (`MyDrive/paying_to_think/`) and the main run saves to `MyDrive/paying_to_think/main/`; run locally, they save to `data/pilot/` and `data/raw/`. After a Colab run, copy `assignment.csv` and `main_results.csv` into `data/raw/` before running the analysis.

In pilot part 2 and the main run, every problem uses its dataset index as its sampling seed (pilot part 1 used greedy decoding, so it needs no seed). The pilot sample uses seed 42, and the main assignment uses seed 2026, so reruns reproduce the same problems and arms. `run_main.py` saves the assignment before running anything, works in chunks of 50 that alternate between arms, and resumes where it stopped if Colab disconnects.
