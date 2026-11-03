"""
Paying to Think: MAIN EXPERIMENT

Follows PREREGISTRATION.md exactly:
    - 1,050 problems randomly sampled from the GSM8K TEST set (seed 2026).
      Pilot problems came from the TRAINING set, so there is no overlap.
    - Random assignment: 350 problems per arm, each problem used in exactly one arm.
    - Budgets (set in pilot part 2): low = 488, medium = 890, unlimited = 4,096 cap.
    - Decoding: temperature 0.6, top_p 0.95, seed = the problem's dataset index.
    - Budget forcing: if thinking hasn't ended at the budget, append
      "</think> Final answer:" and allow 30 more tokens (greedy) for the answer.

Practical features:
    - Saves the assignment BEFORE running anything (assignment.csv).
    - Runs in chunks of 50 and saves after every chunk, so a Colab disconnect
      loses at most one chunk. Re-running the script resumes where it stopped.
    - Chunks alternate between arms, so a partial run stays balanced.
    - Does NOT print accuracy while running (pre-registered stopping rule:
      no looking at results until all 1,050 problems are done).

Run on a machine with an NVIDIA GPU (e.g., Google Colab with a T4 runtime):
    pip install vllm datasets pandas
    python run_main.py
Outputs: assignment.csv, main_results.csv (includes full model output text)
"""

import itertools
import os
import re
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------- settings (pre-registered)
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BUDGETS = {"low": 488, "medium": 890, "unlimited": 4096}
CAP = 4096
FORCE_TOKENS = 30
TEMPERATURE, TOP_P = 0.6, 0.95
N_PER_ARM = 350
ASSIGN_SEED = 2026
CHUNK = 50

# Save to Google Drive when running in Colab, otherwise to a local folder
try:
    from google.colab import drive  # noqa: F401
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/paying_to_think/main"
except ImportError:
    OUT_DIR = os.path.join("data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
ASSIGN_CSV = os.path.join(OUT_DIR, "assignment.csv")
RESULTS_CSV = os.path.join(OUT_DIR, "main_results.csv")


# ---------------------------------------------------------------- helpers (same as pilot part 2)
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


# ---------------------------------------------------------------- assignment
def make_assignment(n_test):
    """Randomly sample 1,050 test problems and assign 350 to each arm. Saved before any run."""
    if os.path.exists(ASSIGN_CSV):
        print("Loaded saved assignment.")
        return pd.read_csv(ASSIGN_CSV)
    rng = np.random.default_rng(ASSIGN_SEED)
    chosen = rng.permutation(n_test)[: N_PER_ARM * len(BUDGETS)]   # random sample, random order
    arms = np.repeat(list(BUDGETS), N_PER_ARM)                      # first 350 -> low, etc.
    df = pd.DataFrame({"orig_idx": chosen, "arm": arms})
    df["budget"] = df["arm"].map(BUDGETS)
    df.to_csv(ASSIGN_CSV, index=False)
    print(f"Saved assignment: {len(df)} problems, {N_PER_ARM} per arm.")
    return df


# ---------------------------------------------------------------- one chunk
def run_chunk(llm, tok, test, chunk):
    """Run one chunk of problems (all in the same arm) and return result rows."""
    budget = int(chunk["budget"].iloc[0])
    arm = chunk["arm"].iloc[0]
    examples = [test[int(i)] for i in chunk["orig_idx"]]
    prompts = [make_prompt(tok, ex["question"]) for ex in examples]
    seeds = [int(i) for i in chunk["orig_idx"]]
    params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budget, seed=s)
              for s in seeds]

    outs = llm.generate([as_tokens(tok, p) for p in prompts], params, use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    gen_tokens = [len(o.outputs[0].token_ids) for o in outs]

    # Budget forcing
    cut = [i for i, t in enumerate(texts) if "</think>" not in t]
    forced_tokens = [0] * len(texts)
    if cut:
        f_prompts = [prompts[i] + texts[i] + "\n</think>\n\nFinal answer:" for i in cut]
        f_outs = llm.generate([as_tokens(tok, p) for p in f_prompts],
                              SamplingParams(temperature=0, max_tokens=FORCE_TOKENS), use_tqdm=False)
        for i, o in zip(cut, f_outs):
            texts[i] += "\n</think>\n\nFinal answer:" + o.outputs[0].text
            forced_tokens[i] = len(o.outputs[0].token_ids)

    cut_set = set(cut)
    rows = []
    for i, ex in enumerate(examples):
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        pred = extract(texts[i])
        rows.append(dict(
            orig_idx=seeds[i], arm=arm, budget=budget,
            gen_tokens=gen_tokens[i], forced_tokens=forced_tokens[i],
            total_tokens=gen_tokens[i] + forced_tokens[i],
            was_cut_off=(i in cut_set), pred=pred, gold=gold,
            correct=(pred is not None and abs(pred - gold) < 1e-6),
            text=texts[i]))
    return rows


# ---------------------------------------------------------------- main
def main():
    test = load_dataset("openai/gsm8k", "main", split="test")
    assign = make_assignment(len(test))

    done = set(pd.read_csv(RESULTS_CSV)["orig_idx"]) if os.path.exists(RESULTS_CSV) else set()
    todo = assign[~assign["orig_idx"].isin(done)]
    print(f"Already done: {len(done)} | still to run: {len(todo)}")
    if todo.empty:
        print("All problems are done. Next step: run analysis.py.")
        return

    # Split each arm into chunks, then alternate arms: low, medium, unlimited, low, ...
    per_arm = [[g.iloc[i:i + CHUNK] for i in range(0, len(g), CHUNK)]
               for _, g in todo.groupby("arm", sort=False)]
    order = [c for group in itertools.zip_longest(*per_arm) for c in group if c is not None]

    # dtype="half" is required on a T4 (it can't run bfloat16)
    llm = LLM(model=MODEL, dtype="half", max_model_len=CAP + 1024,
              gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()

    n_done = len(done)
    for k, chunk in enumerate(order, start=1):
        t0 = time.time()
        rows = run_chunk(llm, tok, test, chunk)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, mode="a",
                                  header=not os.path.exists(RESULTS_CSV), index=False)
        n_done += len(rows)
        print(f"Chunk {k}/{len(order)} | arm = {chunk['arm'].iloc[0]:<9} | "
              f"{len(rows)} problems | {time.time() - t0:.0f}s | total done: {n_done}/{len(assign)}")

    # Completeness check only (no accuracy: pre-registered stopping rule)
    final = pd.read_csv(RESULTS_CSV)
    print("\nProblems completed per arm:")
    print(final["arm"].value_counts().reindex(list(BUDGETS)))
    print(f"\nDone. Results saved to {RESULTS_CSV}. Next step: run analysis.py.")


if __name__ == "__main__":
    main()
