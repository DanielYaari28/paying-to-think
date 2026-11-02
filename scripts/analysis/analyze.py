"""
Paying to Think: STEP 2 - statistical analysis

Runs the analysis plan exactly as pre-registered (Preregistration, Oct 27, 2026),
then a set of clearly labeled EXPLORATORY checks that were not part of the plan.

PRE-REGISTERED (confirmatory)
    1. Main test: chi-square test of independence on the 3x2 table
       (budget arm x correct/incorrect), alpha = 0.05.
    2. Assumptions: independence (each problem used in exactly one arm, by design)
       and every expected cell count >= 5 (otherwise Fisher's exact test).
    3. Post hoc, only if the main test is significant: three pairwise 2x2
       chi-square tests at Bonferroni-adjusted alpha = 0.05 / 3 = 0.0167.
    4. Effect sizes: difference in accuracy between arms, and Cohen's d on the
       0/1 outcome with pooled SD (same formula used in the power analysis).
    5. 95% t-based confidence intervals for each arm's accuracy and mean tokens.
    6. Count of outputs with no extractable number, per arm.

EXPLORATORY (not part of the test; reported separately)
    E1. Unlimited arm: are runs that hit the 4,096-token cap wrong more often?
    E2. Sensitivity check: in the low and medium arms some runs finished thinking
        but ran out of budget while writing the answer, so the answer was cut off
        and no forced answer was requested. We re-run the main test (a) treating
        every such run as correct (the most favorable case for H0) and
        (b) dropping those runs, to see whether the conclusion depends on them.
    E3. Cramer's V as a single effect size for the overall 3x2 association.

Usage (from the repo root, after clean_data.py):
    python scripts/analysis/analyze.py
Outputs: results/*.csv, results/summary.txt, figures/*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CLEAN = os.path.join("data", "clean", "main_results_clean.csv")
RESULTS_DIR = "results"
FIG_DIR = "figures"

ARMS = ["low", "medium", "unlimited"]
BUDGETS = {"low": 488, "medium": 890, "unlimited": 4096}
PAIRS = [("low", "medium"), ("low", "unlimited"), ("medium", "unlimited")]
ALPHA = 0.05
ALPHA_ADJ = ALPHA / len(PAIRS)
COLORS = {"low": "#d95f02", "medium": "#7570b3", "unlimited": "#1b9e77"}

log_lines = []


def log(text=""):
    print(text)
    log_lines.append(str(text))


# ---------------------------------------------------------------- statistics helpers
def t_ci(values, conf=0.95):
    """t-based confidence interval for a mean (pre-registered method)."""
    values = np.asarray(values, dtype=float)
    n, mean = len(values), values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf((1 + conf) / 2, df=n - 1) * se
    return mean, mean - half, mean + half


def cohens_d_binary(p1, p2):
    """Cohen's d on a 0/1 outcome with pooled SD, same formula as the power analysis."""
    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p2 - p1) / pooled_sd


def diff_ci(p1, n1, p2, n2, conf=0.95):
    """95% CI for a difference in two proportions (unpooled standard error)."""
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf((1 + conf) / 2)
    d = p2 - p1
    return d, d - z * se, d + z * se


def contingency(df):
    """3x2 (or 2x2) table: rows = arms, columns = incorrect / correct."""
    table = pd.crosstab(df["arm"], df["correct_int"]).reindex(
        [a for a in ARMS if a in set(df["arm"])])
    table = table.reindex(columns=[0, 1], fill_value=0)
    table.columns = ["incorrect", "correct"]
    return table


def chi_square(table):
    """Chi-square test of independence. No Yates correction, so the 2x2 post hoc
    tests use the same statistic as the main test."""
    chi2, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
    return chi2, p, dof, expected


def cramers_v(chi2, table):
    n = table.values.sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k))


# ---------------------------------------------------------------- analysis sections
def descriptives(df):
    log("=" * 70)
    log("1. DESCRIPTIVE STATISTICS (95% t-based CIs, pre-registered)")
    log("=" * 70)
    rows = []
    for arm in ARMS:
        g = df[df["arm"] == arm]
        acc, acc_lo, acc_hi = t_ci(g["correct_int"])
        tok, tok_lo, tok_hi = t_ci(g["total_tokens"])
        rows.append(dict(
            arm=arm, budget=BUDGETS[arm], n=len(g), n_correct=int(g["correct_int"].sum()),
            accuracy=acc, acc_ci_low=acc_lo, acc_ci_high=acc_hi,
            mean_tokens=tok, tokens_ci_low=tok_lo, tokens_ci_high=tok_hi,
            median_tokens=g["total_tokens"].median(),
            no_answer=int(g["no_answer"].sum()),
            hit_budget_thinking=int(g["hit_budget_thinking"].sum()),
            hit_budget_answer=int(g["hit_budget_answer"].sum()),
        ))
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        log(f"{r.arm:>9} (budget {r.budget:>4}): accuracy {r.accuracy:.3f} "
            f"[{r.acc_ci_low:.3f}, {r.acc_ci_high:.3f}] | mean tokens {r.mean_tokens:,.0f} "
            f"[{r.tokens_ci_low:,.0f}, {r.tokens_ci_high:,.0f}] | no extractable number: {r.no_answer}")
    out.to_csv(os.path.join(RESULTS_DIR, "descriptives.csv"), index=False)
    return out


def main_test(df):
    log("\n" + "=" * 70)
    log("2. MAIN TEST: chi-square of independence, 3x2 table (pre-registered)")
    log("=" * 70)
    table = contingency(df)
    chi2, p, dof, expected = chi_square(table)
    log(table.to_string())
    log(f"\nExpected counts under H0:\n{pd.DataFrame(expected, index=table.index, columns=table.columns).round(1).to_string()}")
    min_exp = expected.min()
    log(f"Assumption check: smallest expected count = {min_exp:.1f} "
        f"({'>= 5, chi-square is valid' if min_exp >= 5 else '< 5, use Fisher exact test instead'})")
    log("Assumption check: independence holds by design (each problem used in exactly one arm).")
    log(f"\nchi2({dof}, N = {table.values.sum()}) = {chi2:.2f}, p = {p:.2e}")
    decision = "REJECT H0" if p < ALPHA else "FAIL TO REJECT H0"
    log(f"Decision at alpha = {ALPHA}: {decision}")
    v = cramers_v(chi2, table)
    log(f"[Exploratory E3] Cramer's V = {v:.3f}")
    table.to_csv(os.path.join(RESULTS_DIR, "contingency_table.csv"))
    pd.DataFrame([dict(test="chi-square of independence (3x2)", chi2=chi2, df=dof, p_value=p,
                       min_expected=min_exp, cramers_v=v, decision=decision)]).to_csv(
        os.path.join(RESULTS_DIR, "main_test.csv"), index=False)
    return p < ALPHA


def post_hoc(df, desc):
    log("\n" + "=" * 70)
    log(f"3. POST HOC: pairwise 2x2 chi-square, Bonferroni alpha = {ALPHA_ADJ:.4f} (pre-registered)")
    log("=" * 70)
    acc = desc.set_index("arm")
    rows = []
    for a, b in PAIRS:
        sub = df[df["arm"].isin([a, b])]
        chi2, p, _, expected = chi_square(contingency(sub))
        pa, pb, na, nb = acc.loc[a, "accuracy"], acc.loc[b, "accuracy"], acc.loc[a, "n"], acc.loc[b, "n"]
        d_acc, lo, hi = diff_ci(pa, na, pb, nb)
        d = cohens_d_binary(pa, pb)
        sig = p < ALPHA_ADJ
        rows.append(dict(comparison=f"{a} vs {b}", acc_a=pa, acc_b=pb,
                         acc_difference=d_acc, diff_ci_low=lo, diff_ci_high=hi,
                         cohens_d=d, chi2=chi2, p_value=p,
                         p_bonferroni=min(p * len(PAIRS), 1.0),
                         significant_at_adj_alpha=sig, min_expected=expected.min()))
        log(f"{a:>6} vs {b:<9}: {pa:.3f} -> {pb:.3f} | diff = {d_acc:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] | d = {d:.2f} | chi2(1) = {chi2:.2f}, p = {p:.2e} "
            f"| {'significant' if sig else 'not significant'}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "pairwise.csv"), index=False)
    log("Cohen's d guide: 0.2 small, 0.5 medium, 0.8 large.")


def exploratory_cap(df):
    log("\n" + "=" * 70)
    log("E1. EXPLORATORY: unlimited arm, hit the 4,096 cap vs finished on its own")
    log("=" * 70)
    u = df[df["arm"] == "unlimited"]
    g = u.groupby("hit_budget_thinking")["correct_int"].agg(["size", "sum", "mean"])
    g.index = g.index.map({False: "finished on its own", True: "hit the cap"})
    log(g.rename(columns={"size": "n", "sum": "correct", "mean": "accuracy"}).round(3).to_string())
    table = pd.crosstab(u["hit_budget_thinking"], u["correct_int"]).values
    _, p = stats.fisher_exact(table)
    log(f"Fisher's exact test (small cell counts): p = {p:.2e}")
    log("Interpretation: runs that hit the cap seem to be stuck rather than thinking productively.")


def sensitivity(df):
    log("\n" + "=" * 70)
    log("E2. EXPLORATORY SENSITIVITY: runs whose ANSWER was cut off by the budget")
    log("=" * 70)
    flagged = df["hit_budget_answer"]
    log(df[flagged].groupby("arm")["correct_int"].agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "accuracy"}).reindex(ARMS).fillna(0).round(3).to_string())
    log("These runs finished thinking but the budget ended mid-answer, so budget forcing\n"
        "never triggered and the extracted number is often not the model's answer.")

    scenarios = {
        "as pre-registered": df,
        "flagged runs counted as correct (best case for H0)": df.assign(
            correct_int=np.where(flagged, 1, df["correct_int"])),
        "flagged runs dropped": df[~flagged],
    }
    rows = []
    for name, d in scenarios.items():
        table = contingency(d)
        chi2, p, dof, _ = chi_square(table)
        accs = (table["correct"] / table.sum(axis=1)).reindex(ARMS)
        pair_p = {}
        for a, b in PAIRS:
            pair_p[f"p_{a}_vs_{b}"] = chi_square(contingency(d[d["arm"].isin([a, b])]))[1]
        rows.append(dict(scenario=name, n=len(d), acc_low=accs["low"], acc_medium=accs["medium"],
                         acc_unlimited=accs["unlimited"], chi2=chi2, p_value=p, **pair_p))
        log(f"\n{name}: n = {len(d)} | accuracy low {accs['low']:.3f}, medium {accs['medium']:.3f}, "
            f"unlimited {accs['unlimited']:.3f}\n   omnibus chi2({dof}) = {chi2:.2f}, p = {p:.2e} | "
            + ", ".join(f"{k[2:].replace('_', ' ')} p = {v:.2e}" for k, v in pair_p.items()))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"), index=False)


# ---------------------------------------------------------------- figures
def fig_accuracy(desc):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(ARMS))
    err = [desc["accuracy"] - desc["acc_ci_low"], desc["acc_ci_high"] - desc["accuracy"]]
    ax.bar(x, desc["accuracy"], yerr=err, capsize=6, color=[COLORS[a] for a in ARMS], width=0.6)
    for i, r in desc.iterrows():
        ax.text(i, r["acc_ci_high"] + 0.02, f"{r['accuracy']:.1%}", ha="center", fontsize=11)
    ax.set_xticks(x, [f"{a.capitalize()}\n({BUDGETS[a]:,} tokens)" for a in ARMS])
    ax.set_ylabel("Accuracy on GSM8K (n = 350 per arm)")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy rises with thinking budget\n(error bars: 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_accuracy_by_budget.png"), dpi=200)
    plt.close(fig)


def fig_tokens(df):
    """Natural thinking length (unlimited arm) with the low/medium budgets marked."""
    u = df.loc[df["arm"] == "unlimited", "gen_tokens"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(u, bins=np.linspace(0, 4200, 50), color=COLORS["unlimited"], alpha=0.8)
    top = ax.get_ylim()[1]
    for arm, height in [("low", 0.92), ("medium", 0.80)]:
        share = (u > BUDGETS[arm]).mean()
        ax.axvline(BUDGETS[arm], color=COLORS[arm], ls="--", lw=1.5)
        ax.text(BUDGETS[arm] + 60, top * height,
                f"{arm} budget ({BUDGETS[arm]}): would cut {share:.0%}", color=COLORS[arm], fontsize=9)
    ax.set_xlabel("Tokens generated with no practical limit (unlimited arm)")
    ax.set_ylabel("Number of problems")
    ax.set_title("How long the model naturally thinks (spike at right = hit the 4,096 cap)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_natural_thinking_length.png"), dpi=200)
    plt.close(fig)


def fig_outcomes(df):
    """Stacked bars: how each run ended, split into correct / incorrect."""
    ending = np.select([df["hit_budget_thinking"], df["hit_budget_answer"]],
                       ["budget hit while thinking (answer forced)", "budget hit while answering"],
                       "finished within budget")
    d = df.assign(ending=ending)
    order = ["finished within budget", "budget hit while thinking (answer forced)", "budget hit while answering"]
    shades = {"finished within budget": "#4c72b0",
              "budget hit while thinking (answer forced)": "#dd8452",
              "budget hit while answering": "#c44e52"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(ARMS))
    bottom = np.zeros(len(ARMS))
    for end in order:
        for correct, hatch in [(1, ""), (0, "///")]:
            vals = np.array([((d["arm"] == a) & (d["ending"] == end) & (d["correct_int"] == correct)).sum()
                             for a in ARMS])
            ax.bar(x, vals, bottom=bottom, color=shades[end], hatch=hatch, edgecolor="white", width=0.6,
                   label=f"{end}: {'correct' if correct else 'incorrect'}")
            bottom += vals
    ax.set_xticks(x, [a.capitalize() for a in ARMS])
    ax.set_ylabel("Number of problems")
    ax.set_title("How each run ended (hatched = incorrect)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_run_outcomes.png"), dpi=200)
    plt.close(fig)


def fig_tradeoff(desc):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(desc["mean_tokens"], desc["accuracy"],
                yerr=[desc["accuracy"] - desc["acc_ci_low"], desc["acc_ci_high"] - desc["accuracy"]],
                fmt="-o", color="#333333", capsize=5)
    for _, r in desc.iterrows():
        ax.annotate(r["arm"], (r["mean_tokens"], r["accuracy"]), textcoords="offset points",
                    xytext=(8, -12), color=COLORS[r["arm"]], fontsize=11)
    ax.set_xlabel("Mean tokens used per problem")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost vs. accuracy: diminishing returns")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_cost_vs_accuracy.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(CLEAN)

    desc = descriptives(df)
    significant = main_test(df)
    if significant:
        post_hoc(df, desc)
    else:
        log("\nMain test not significant: post hoc tests are not run (pre-registered rule).")
    exploratory_cap(df)
    sensitivity(df)

    fig_accuracy(desc)
    fig_tokens(df)
    fig_outcomes(df)
    fig_tradeoff(desc)

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nSaved tables to {RESULTS_DIR}/ and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
