"""
Paying to Think: STEP 1 - data cleaning and validation

What this script does:
    1. Loads the raw experiment output (data/raw/main_results.csv) and the saved
       random assignment (data/raw/assignment.csv).
    2. Runs integrity checks that must all pass before any analysis:
         - 350 problems per arm, 1,050 total
         - no problem appears twice
         - every result matches its pre-saved arm and budget (no mix-ups)
    3. Adds derived columns used in the analysis (nothing is re-scored; the
       pre-registered `correct` column is kept exactly as produced):
         - correct_int          0/1 version of `correct`
         - no_answer            no number could be extracted (counted as incorrect)
         - extraction_method    which pre-registered rule produced the answer
                                (boxed / label / last_number)
         - hit_budget_thinking  budget ran out mid-thinking -> answer was forced
         - hit_budget_answer    model finished thinking but the budget ran out
                                while it was writing the answer (no forcing happens
                                in this case, so the answer may be truncated)
    4. Writes data/clean/main_results_clean.csv WITHOUT the long raw model text
       (the raw text stays in data/raw/ for anyone who wants to audit outputs).

Usage (from the repo root):
    pip install -r requirements.txt
    python scripts/analysis/clean_data.py
"""

import os
import re
import sys

import pandas as pd

RAW_RESULTS = os.path.join("data", "raw", "main_results.csv")
RAW_ASSIGN = os.path.join("data", "raw", "assignment.csv")
CLEAN_OUT = os.path.join("data", "clean", "main_results_clean.csv")

ARMS = ["low", "medium", "unlimited"]
BUDGETS = {"low": 488, "medium": 890, "unlimited": 4096}
N_PER_ARM = 350


def extraction_method(text):
    """Mirror the pre-registered extraction rule and report which branch fired."""
    answer_part = str(text).split("</think>")[-1]
    if re.search(r"\\boxed\{([^}]*)\}", answer_part):
        return "boxed"
    if "Final answer:" in answer_part:
        return "label"
    return "last_number"


def check(condition, message):
    """Stop immediately if an integrity check fails."""
    if not condition:
        sys.exit(f"CHECK FAILED: {message}")
    print(f"  ok - {message}")


def main():
    raw = pd.read_csv(RAW_RESULTS)
    assign = pd.read_csv(RAW_ASSIGN)

    print("Integrity checks")
    check(len(raw) == N_PER_ARM * len(ARMS), f"{len(raw)} rows (expected {N_PER_ARM * len(ARMS)})")
    check(raw["orig_idx"].is_unique, "no problem appears more than once")
    counts = raw["arm"].value_counts()
    check(all(counts.get(a, 0) == N_PER_ARM for a in ARMS), f"{N_PER_ARM} problems in every arm")
    check(all(raw["budget"] == raw["arm"].map(BUDGETS)), "every row's budget matches its arm")

    merged = raw.merge(assign, on="orig_idx", suffixes=("", "_assigned"), how="left")
    check(merged["arm_assigned"].notna().all(), "every result appears in the saved assignment")
    check((merged["arm"] == merged["arm_assigned"]).all(), "every result ran in its pre-assigned arm")

    clean = raw.copy()
    clean["correct_int"] = clean["correct"].astype(int)
    clean["no_answer"] = clean["pred"].isna()
    clean["extraction_method"] = clean["text"].map(extraction_method)
    clean["hit_budget_thinking"] = clean["was_cut_off"]
    clean["hit_budget_answer"] = (~clean["was_cut_off"]) & (clean["gen_tokens"] >= clean["budget"])

    # Sanity: an extraction failure must always be scored incorrect
    check(not (clean["no_answer"] & clean["correct"]).any(), "no-answer outputs are all scored incorrect")

    clean = clean.drop(columns=["text"])
    os.makedirs(os.path.dirname(CLEAN_OUT), exist_ok=True)
    clean.to_csv(CLEAN_OUT, index=False)

    print("\nPer-arm overview (counts)")
    print(clean.groupby("arm")[["no_answer", "hit_budget_thinking", "hit_budget_answer"]]
          .sum().reindex(ARMS).to_string())
    print(f"\nWrote {CLEAN_OUT} ({len(clean)} rows, raw text column dropped)")


if __name__ == "__main__":
    main()
