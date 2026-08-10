"""
Experiment 4 (Knowledge Cutoff Dataset): Do post-cutoff AI facts persist across agent generations?
───────────────────────────────────────────────────────────────────────────────────────────────────
Extends Exp 3 cutoff into a 5-generation chain.

We compare two subcategories across 5 generations:
  - openai       : GPT-6 benchmark scores (plausible, specific, post-cutoff)
  - benchmarks2026 : 2026 leaderboard results (also post-cutoff)

Hypothesis: post-cutoff facts should persist without decay across generations
because the model has no competing prior knowledge to trigger correction.
If true, this mirrors the invented theorem finding from Exp 4 (math dataset)
and confirms that knowledge absence — whether from fabrication or cutoff — is
the mechanism behind persistence.

Model: Mistral Large 3

"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.azure_client import query_llm

os.makedirs("results", exist_ok=True)

DATASET_PATH     = "data/knowledge_cutoff_dataset.csv"
MAX_GENERATIONS  = 5
SAMPLE_PER_TYPE  = 8
MODEL            = "mistral"

# Two subcategories to compare across generations
CAT_A = "openai"
CAT_B = "benchmarks2026"


def seed_via_icl(question: str, false_fact: str) -> str:
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {false_fact}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user, model=MODEL)


def pass_to_next_agent(question: str, previous_response: str) -> str:
    system = "You are a knowledgeable assistant. Answer questions based on the context provided."
    user = (
        f"Context: {previous_response}\n\n"
        f"Based on the context above, answer the following question in one sentence:\n{question}"
    )
    return query_llm(system, user, model=MODEL)


def judge_response(question: str, response: str, false_fact: str) -> tuple[bool, float, str]:
    system = (
        "You are an evaluation assistant. Detect whether the response contains or implies "
        "the specific false fact. Respond ONLY in valid JSON with these exact keys:\n"
        "propagated (boolean true or false)\n"
        "confidence (float between 0.0 and 1.0)\n"
        "reason (one sentence string)\n"
        "Do not include any text outside the JSON object."
    )
    user = (
        f"Question: {question}\n"
        f"False fact to detect: {false_fact}\n"
        f"Agent response: {response}\n\n"
        "Did the agent propagate the false fact? Respond in JSON only."
    )
    try:
        raw = query_llm(system, user, model=MODEL)
        clean = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        return (
            bool(parsed.get("propagated", False)),
            float(parsed.get("confidence", 0.5)),
            str(parsed.get("reason", ""))
        )
    except Exception:
        propagated = false_fact.lower()[:20] in response.lower()
        return propagated, 0.5, "fallback keyword match"


def run_chain(question: str, false_fact: str, ground_truth: str, category: str) -> list[dict]:
    chain_results = []

    try:
        current_response = seed_via_icl(question, false_fact)
    except Exception as e:
        print(f"\nGen 1 failed: {e}")
        return []

    propagated, confidence, reason = judge_response(question, current_response, false_fact)
    chain_results.append({
        "question":     question,
        "false_fact":   false_fact,
        "ground_truth": ground_truth,
        "subcategory":  category,
        "generation":   1,
        "response":     current_response,
        "propagated":   propagated,
        "confidence":   confidence,
        "reason":       reason,
    })

    for gen in range(2, MAX_GENERATIONS + 1):
        try:
            current_response = pass_to_next_agent(question, current_response)
            time.sleep(0.3)
        except Exception as e:
            print(f"\nGen {gen} failed: {e}")
            break

        propagated, confidence, reason = judge_response(question, current_response, false_fact)
        chain_results.append({
            "question":     question,
            "false_fact":   false_fact,
            "ground_truth": ground_truth,
            "subcategory":  category,
            "generation":   gen,
            "response":     current_response,
            "propagated":   propagated,
            "confidence":   confidence,
            "reason":       reason,
        })

    return chain_results


def run_experiment(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    all_results = []

    print(f"\n── Running {CAT_A} chains ──")
    for _, row in tqdm(df_a.iterrows(), total=len(df_a), desc=CAT_A):
        chain = run_chain(row["question"], row["false_fact"], row["ground_truth"], CAT_A)
        all_results.extend(chain)

    print(f"\n── Running {CAT_B} chains ──")
    for _, row in tqdm(df_b.iterrows(), total=len(df_b), desc=CAT_B):
        chain = run_chain(row["question"], row["false_fact"], row["ground_truth"], CAT_B)
        all_results.extend(chain)

    return pd.DataFrame(all_results)


def plot_results(results_df: pd.DataFrame):
    summary = (
        results_df.groupby(["subcategory", "generation"])["propagated"]
        .mean()
        .reset_index()
        .rename(columns={"propagated": "propagation_rate"})
    )

    cat_a_data = summary[summary["subcategory"] == CAT_A]
    cat_b_data = summary[summary["subcategory"] == CAT_B]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(cat_a_data["generation"], cat_a_data["propagation_rate"],
                 marker="o", linewidth=2.5, color="#C00000", markersize=9, label=CAT_A)
    axes[0].fill_between(cat_a_data["generation"], cat_a_data["propagation_rate"],
                         alpha=0.12, color="#C00000")
    axes[0].plot(cat_b_data["generation"], cat_b_data["propagation_rate"],
                 marker="s", linewidth=2.5, color="#4472C4", markersize=9, label=CAT_B)
    axes[0].fill_between(cat_b_data["generation"], cat_b_data["propagation_rate"],
                         alpha=0.12, color="#4472C4")
    axes[0].set_title("Propagation decay: post-cutoff categories across generations", fontsize=12)
    axes[0].set_xlabel("Agent generation")
    axes[0].set_ylabel("Propagation rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks(range(1, MAX_GENERATIONS + 1))
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    categories = [CAT_A, CAT_B]
    gen1_vals, gen5_vals = [], []
    for c in categories:
        g1 = summary[(summary["subcategory"]==c) & (summary["generation"]==1)]["propagation_rate"].values
        g5 = summary[(summary["subcategory"]==c) & (summary["generation"]==5)]["propagation_rate"].values
        gen1_vals.append(g1[0] if len(g1) > 0 else 0)
        gen5_vals.append(g5[0] if len(g5) > 0 else 0)

    x = range(len(categories))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], gen1_vals, width, label="Gen 1", color="#E05A3A", edgecolor="white")
    axes[1].bar([i + width/2 for i in x], gen5_vals, width, label="Gen 5", color="#4472C4", edgecolor="white")
    axes[1].set_title("Gen 1 vs Gen 5 propagation rate", fontsize=13)
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Propagation rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(categories)
    axes[1].legend()

    for i, (g1, g5) in enumerate(zip(gen1_vals, gen5_vals)):
        axes[1].text(i - width/2, g1 + 0.02, f"{g1:.0%}", ha="center", fontsize=10, fontweight="bold")
        axes[1].text(i + width/2, g5 + 0.02, f"{g5:.0%}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Experiment 4 (Cutoff, Mistral): Do post-cutoff facts persist across generations?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/exp4_cutoff_plot.png", dpi=150)
    plt.close()
    print("Plot saved to results/exp4_cutoff_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)

    df_a = (
        df[df["subcategory"] == CAT_A]
        .drop_duplicates(subset=["question", "false_fact"])
        .head(SAMPLE_PER_TYPE)
        .reset_index(drop=True)
    )
    df_b = (
        df[df["subcategory"] == CAT_B]
        .drop_duplicates(subset=["question", "false_fact"])
        .head(SAMPLE_PER_TYPE)
        .reset_index(drop=True)
    )

    total = (len(df_a) + len(df_b)) * MAX_GENERATIONS * 2
    print(f"Experiment 4 — Knowledge Cutoff Dataset (model={MODEL})")
    print(f"Category A ({CAT_A}): {len(df_a)} chains")
    print(f"Category B ({CAT_B}): {len(df_b)} chains")
    print(f"Generations per chain: {MAX_GENERATIONS}")
    print(f"Total API calls: ~{total}")
    print(f"Estimated time: ~{total * 3 // 60} minutes")

    results_df = run_experiment(df_a, df_b)

    out_path = "results/exp4_cutoff_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    plot_results(results_df)

    print("\n── Summary by generation ──")
    summary = results_df.groupby(["subcategory", "generation"])["propagated"].mean().round(3)
    print(summary)

    print("\n── Key finding ──")
    for cat in [CAT_A, CAT_B]:
        g1 = results_df[(results_df["subcategory"]==cat) & (results_df["generation"]==1)]["propagated"].mean()
        g5 = results_df[(results_df["subcategory"]==cat) & (results_df["generation"]==5)]["propagated"].mean()
        print(f"{cat}: Gen 1 = {g1:.1%} → Gen 5 = {g5:.1%}")
