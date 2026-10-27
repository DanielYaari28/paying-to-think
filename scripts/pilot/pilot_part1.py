"""
Paying to Think: PILOT PART 1 (feasibility pilot)

What this script did:
    Ran 60 GSM8K *training* problems through DeepSeek-R1-Distill-Qwen-1.5B with
    greedy decoding and a 4,096-token cap, one problem at a time (Hugging Face
    transformers), to check that:
      - the model runs at a usable speed on a T4 GPU,
      - budget forcing works (cutting off thinking and forcing an answer),
      - answers can be extracted reliably,
    and to see how long the model naturally "thinks".

Key finding that changed the design:
    7 of 60 runs hit the 4,096-token cap, apparently stuck repeating themselves
    under greedy decoding. The main experiment therefore uses the developer's
    recommended sampling settings (temperature 0.6, top_p 0.95) with a fixed seed
    per problem. The budgets and power analysis come from pilot part 2, not this one.

Note: this run saved results only, not the full model output text.

Run on a machine with an NVIDIA GPU (e.g., Google Colab with a T4 runtime):
    pip install transformers datasets accelerate pandas
    python pilot_part1.py
Output: pilot_natural.csv
"""

import os
import re
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- settings
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BUDGET = 4096               # effectively unlimited
FORCE_TOKENS = 30           # tokens allowed for a forced final answer
N_PILOT = 60
SHUFFLE_SEED = 42

# Save to Google Drive when running in Colab, otherwise to a local folder
try:
    from google.colab import drive  # noqa: F401
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/paying_to_think"
except ImportError:
    OUT_DIR = os.path.join("data", "pilot")
os.makedirs(OUT_DIR, exist_ok=True)
PILOT_CSV = os.path.join(OUT_DIR, "pilot_natural.csv")


# ---------------------------------------------------------------- model + data
def load_model():
    assert torch.cuda.is_available(), "No GPU found. In Colab: Runtime > Change runtime type > T4 GPU"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
    print("Model on:", model.device)
    return tok, model


def run(tok, model, question, budget):
    """Generate up to `budget` tokens; if thinking never ends, force a final answer."""
    msgs = [{"role": "user", "content": question + "\nPut your final answer after 'Final answer:'."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=budget, do_sample=False)  # greedy
    gen = out[0][ids.input_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    n_gen, forced_tokens = len(gen), 0

    if "</think>" not in text:  # budget hit: cut off thinking and force an answer
        forced = prompt + text + "\n</think>\n\nFinal answer:"
        ids2 = tok(forced, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out2 = model.generate(**ids2, max_new_tokens=FORCE_TOKENS, do_sample=False)
        gen2 = out2[0][ids2.input_ids.shape[1]:]
        forced_tokens = len(gen2)
        text += "\n</think>\n\nFinal answer:" + tok.decode(gen2, skip_special_tokens=True)
    return text, n_gen, forced_tokens


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


# ---------------------------------------------------------------- main
def main():
    tok, model = load_model()
    data = load_dataset("openai/gsm8k", "main", split="train")
    pilot = data.shuffle(seed=SHUFFLE_SEED).select(range(N_PILOT))

    # Resume support: skip problems already saved
    done = set(pd.read_csv(PILOT_CSV)["idx"]) if os.path.exists(PILOT_CSV) else set()

    for i, ex in enumerate(pilot):
        if i in done:
            continue
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        t = time.time()
        text, n, f = run(tok, model, ex["question"], BUDGET)
        pred = extract(text)
        row = dict(idx=i, budget=BUDGET, tokens=n, forced_tokens=f, hit_cap=(f > 0),
                   pred=pred, gold=gold,
                   correct=(pred is not None and abs(pred - gold) < 1e-6),
                   seconds=round(time.time() - t, 1))
        pd.DataFrame([row]).to_csv(PILOT_CSV, mode="a", header=not os.path.exists(PILOT_CSV), index=False)
        print(row)

    df = pd.read_csv(PILOT_CSV)
    print(df["tokens"].describe())
    print("25th percentile:", df["tokens"].quantile(0.25), "| median:", df["tokens"].median())
    print("accuracy:", df["correct"].mean(), "| hit the 4,096 cap:", df["hit_cap"].sum())


if __name__ == "__main__":
    main()
