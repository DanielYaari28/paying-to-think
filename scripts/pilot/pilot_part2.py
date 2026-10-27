"""
Paying to Think: PILOT PART 2 (budget setting + power analysis)

What this script does:
    Uses the same 60 GSM8K *training* problems as pilot part 1, now with the
    developer's recommended sampling settings (temperature 0.6, top_p 0.95,
    fixed seed per problem) and the vLLM engine.

    Step A: runs every problem with a 4,096-token cap to measure natural
            generation length, then sets the budgets by the pre-registered rule:
              LOW    = 25th percentile of natural length
              MEDIUM = median of natural length
    Step B: runs the same problems at the LOW and MEDIUM budgets.
    Step C: reports accuracy per arm and runs the power analysis
            (correct/incorrect as 0/1, Cohen's d with pooled SD,
            statsmodels TTestIndPower, power = 0.80, Bonferroni alpha = 0.05/3).

    Results from this run: LOW = 488 tokens, MEDIUM = 890 tokens;
    pilot accuracy 0.667 / 0.783 / 0.883; required n = 303 per arm (rounded up to 350).

    Pilot problem IDs are saved so the main experiment can exclude them.

Run on a machine with an NVIDIA GPU (e.g., Google Colab with a T4 runtime):
    pip install vllm datasets pandas statsmodels
    python pilot_part2.py
Outputs: pilot_ids.csv, pilot2_unlimited.csv, pilot2_low.csv, pilot2_medium.csv,
         pilot2_summary.csv (all include the full model output text)
"""

import os
import re
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from statsmodels.stats.power import TTestIndPower
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------- settings
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CAP = 4096                      # safety cap for the "unlimited" arm
FORCE_TOKENS = 30               # tokens allowed for a forced final answer
TEMPERATURE, TOP_P = 0.6, 0.95  # developer's recommended settings
N_PILOT, PILOT_SHUFFLE_SEED = 60, 42
ALPHA, POWER, N_COMPARISONS = 0.05, 0.80, 3

# Save to Google Drive when running in Colab, otherwise to a local folder
try:
    from google.colab import drive  # noqa: F401
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/paying_to_think"
except ImportError:
    OUT_DIR = os.path.join("data", "pilot")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def make_prompt(tok, question):
    msgs = [{"role": "user", "content": question + "\nPut your final answer after 'Final answer:'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def as_tokens(tok, text):
    """Pass token IDs directly so no duplicate start-of-text token is added."""
    return {"prompt_token_ids": tok.encode(text, add_special_tokens=False)}


def extract(text):
    """Pull the numeric answer from the text after </think>."""
    ans = text.split("</think>")[-1]
    m = re.search(r"\\boxed\{([^}]*)\}", ans)
    if m:
        seg, pick = m.group(1), 0                       # \boxed{answer}
    elif "Final answer:" in ans:
        seg, pick = ans.split("Final answer:")[-1], 0   # first number after the label
    else:
        seg, pick = ans, -1                             # otherwise the last number
    nums = re.findall(r"-?\d[\d,]*\.?\d*", seg)
    if not nums:
        return None
    try:
        return float(nums[pick].replace(",", "").rstrip("."))
    except ValueError:
        return None


def run_arm(llm, tok, problems, budget, arm_name):
    """Run every problem at one thinking budget; force an answer if the budget is hit."""
    prompts = [make_prompt(tok, q) for q in problems["question"]]
    seeds = [int(s) for s in problems["orig_idx"]]      # one fixed seed per problem
    params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budget, seed=s)
              for s in seeds]
    t0 = time.time()
    outs = llm.generate([as_tokens(tok, p) for p in prompts], params)
    texts = [o.outputs[0].text for o in outs]
    gen_tokens = [len(o.outputs[0].token_ids) for o in outs]

    # Budget forcing: if thinking never ended, cut it off and ask for the answer
    cut = [i for i, t in enumerate(texts) if "</think>" not in t]
    forced_tokens = [0] * len(texts)
    if cut:
        f_prompts = [prompts[i] + texts[i] + "\n</think>\n\nFinal answer:" for i in cut]
        f_outs = llm.generate([as_tokens(tok, p) for p in f_prompts],
                              SamplingParams(temperature=0, max_tokens=FORCE_TOKENS))
        for i, o in zip(cut, f_outs):
            texts[i] += "\n</think>\n\nFinal answer:" + o.outputs[0].text
            forced_tokens[i] = len(o.outputs[0].token_ids)

    golds = [float(a.split("####")[-1].strip().replace(",", "")) for a in problems["answer"]]
    preds = [extract(t) for t in texts]
    cut_set = set(cut)
    df = pd.DataFrame({
        "orig_idx": seeds, "arm": arm_name, "budget": budget,
        "gen_tokens": gen_tokens, "forced_tokens": forced_tokens,
        "was_cut_off": [i in cut_set for i in range(len(texts))],
        "pred": preds, "gold": golds,
        "correct": [p is not None and abs(p - g) < 1e-6 for p, g in zip(preds, golds)],
        "text": texts,                                   # raw output, kept for the repo
    })
    print(f"{arm_name} (budget={budget}): {time.time() - t0:.0f}s | "
          f"accuracy={df.correct.mean():.3f} | cut off={df.was_cut_off.sum()} of {len(df)}")
    return df


def run_or_load(llm, tok, problems, budget, arm_name, path):
    """Load saved results if they exist, otherwise run the arm and save it."""
    if os.path.exists(path):
        print(f"Loaded saved {arm_name}-arm results.")
        return pd.read_csv(path)
    df = run_arm(llm, tok, problems, budget, arm_name)
    df.to_csv(path, index=False)
    return df


def n_needed(p1, p2, alpha):
    """Treat correct/incorrect as 0/1, compute Cohen's d with pooled SD, solve for n per arm."""
    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled_sd == 0 or p1 == p2:
        return 0.0, float("inf")
    d = abs(p1 - p2) / pooled_sd
    n = TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=POWER, alternative="two-sided")
    return d, n


# ---------------------------------------------------------------- main
def main():
    # dtype="half" is required on a T4 (it can't run bfloat16)
    llm = LLM(model=MODEL, dtype="half", max_model_len=CAP + 1024,
              gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()

    data = load_dataset("openai/gsm8k", "main", split="train")
    data = data.add_column("orig_idx", list(range(len(data))))
    # Same shuffle as pilot part 1, so these are the same 60 problems
    pilot = data.shuffle(seed=PILOT_SHUFFLE_SEED).select(range(N_PILOT))
    pd.DataFrame({"orig_idx": pilot["orig_idx"]}).to_csv(
        os.path.join(OUT_DIR, "pilot_ids.csv"), index=False)
    print("Pilot problems:", len(pilot))

    # Step A: natural length (unlimited arm) -> set budgets
    unl = run_or_load(llm, tok, pilot, CAP, "unlimited",
                      os.path.join(OUT_DIR, "pilot2_unlimited.csv"))
    print(unl.gen_tokens.describe())
    print("Finished thinking on their own:", (~unl.was_cut_off).sum(), "of", len(unl))
    low = int(round(unl.gen_tokens.quantile(0.25)))   # pre-registered rule: 25th percentile
    med = int(round(unl.gen_tokens.quantile(0.50)))   # pre-registered rule: median
    print(f"\nBUDGETS -> low = {low} tokens, medium = {med} tokens, unlimited = {CAP} cap")

    # Step B: accuracy at the low and medium budgets
    results = [unl]
    for name, budget in [("low", low), ("medium", med)]:
        results.append(run_or_load(llm, tok, pilot, budget, name,
                                   os.path.join(OUT_DIR, f"pilot2_{name}.csv")))
    allp = pd.concat(results, ignore_index=True)

    # Step C: summary + power analysis (course Chapter 4 method)
    summary = (allp.groupby("arm")
               .agg(budget=("budget", "first"), accuracy=("correct", "mean"),
                    mean_tokens=("gen_tokens", "mean"), median_tokens=("gen_tokens", "median"),
                    cut_off=("was_cut_off", "sum"))
               .reindex(["low", "medium", "unlimited"]))
    print(summary.round(3))
    summary.to_csv(os.path.join(OUT_DIR, "pilot2_summary.csv"))

    acc = summary["accuracy"]
    print(f"\nThree-arm plan (Bonferroni alpha = {ALPHA}/{N_COMPARISONS} = {ALPHA / N_COMPARISONS:.4f}):")
    for a, b in [("low", "unlimited"), ("medium", "unlimited"), ("low", "medium")]:
        d, n = n_needed(acc[a], acc[b], ALPHA / N_COMPARISONS)
        print(f"  {a:>6} vs {b:<9}: acc {acc[a]:.3f} vs {acc[b]:.3f} | d = {d:.3f} | n per arm = {n:,.0f}")

    d, n = n_needed(acc["low"], acc["unlimited"], ALPHA)
    print(f"\nTwo-arm backup (low vs unlimited, alpha = {ALPHA}): d = {d:.3f} | n per arm = {n:,.0f}")
    print("\nNote: these come from only 60 pilot problems, so the effect sizes are rough estimates.")


if __name__ == "__main__":
    main()
